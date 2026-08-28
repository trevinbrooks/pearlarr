"""The engine-facing strategy contracts: the hooks a per-arr strategy implements."""

from abc import ABC, abstractmethod

from .manual_import import AttemptKind, ImportProbe, ImportProgress, PendingImport
from .mappings import MappingEntry
from .seadex_types import ArrItem, HistoryRecord, ProgressSink


class ImportCompleter(ABC):
    """The strategy hooks the engine drives after a download completes, deliberately non-generic."""

    @abstractmethod
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt: AttemptKind,
    ) -> ImportProbe:
        """Reconcile one completed download with the arr (one poll).

        A manual import assigns with *our* file->episode mapping, never the arr's title parse.
        """

    @abstractmethod
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only "files inserted" count for the wait cockpit's bar.

        MUST NOT refresh downloads, read the queue, or issue commands.
        """

    def close_tracked(self, pending: PendingImport) -> None:
        """Dismiss the arr's leftover queue entry for a fully imported torrent.

        Sonarr auto-closes a tracked download only when ONE import covers the grab's full episode count.
        """

        del pending

    @property
    @abstractmethod
    def supports_blocking_monitor(self) -> bool:
        """Whether the engine may run the end-of-run waiting monitor."""


class ArrSync[ItemT: ArrItem](ImportCompleter):
    """An Arr-specific sync strategy the run machinery drives: the Arr REST client plus its domain logic."""

    @abstractmethod
    def get_items(self) -> list[ItemT]:
        """Every Arr item to consider this run (also the run-start hook)."""

    @abstractmethod
    def filter_to_single(self, items: list[ItemT], item_id: int) -> list[ItemT]:
        """Narrow the item list to the single external id `item_id`."""

    @abstractmethod
    def item_anilist_ids(
        self,
        item: ItemT,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve the AniList ids mapped to one Arr item."""

    @property
    @abstractmethod
    def warms_episodes(self) -> bool:
        """Whether `prefetch_episodes` does real work."""

    @abstractmethod
    def prefetch_episodes(self, items: list[ItemT], *, progress: ProgressSink | None = None) -> int:
        """Warm per-item network caches concurrently before the scan loop, returning how many were attempted."""

    @abstractmethod
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Arr history records since `date`, or None on failure."""

    @abstractmethod
    def process_al_id(
        self,
        item: ItemT,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for one Arr item, returning True if it grabbed."""

    @abstractmethod
    def pending_import_series_id(self, item: ItemT) -> int | None:
        """The Arr series id whose carried-over pending records this item owns.

        None skips the engine's per-item pending snapshot.
        """
