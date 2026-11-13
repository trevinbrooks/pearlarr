"""The strategy contracts the arr run drives: the hooks a per-arr strategy implements."""

from abc import ABC, abstractmethod

from .arr_models import ArrItem, HistoryRecord
from .mappings import TitleMapping
from .pending_import import PendingImport
from .progress import ProgressCallback
from .wait_types import AttemptKind, AttemptResult, EffectStatus, ImportProgress


class ImportCompleter(ABC):
    """The strategy hooks the wait pass drives after a torrent completes, deliberately non-generic."""

    @abstractmethod
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt_kind: AttemptKind,
    ) -> AttemptResult:
        """Reconcile one completed torrent with the arr (one attempt).

        An auto import goes by pearlarr's file map, never the arr's parse of the names.
        The attempt result carries any files the arr's history put on episodes, for the wait pass to persist onto the
        pending import.
        """

    @abstractmethod
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only imported-files count for the wait region's bar.

        MUST NOT refresh downloads, read the queue, or issue commands.
        """

    def remove_from_queue(self, pending: PendingImport) -> EffectStatus:
        """Dismiss the arr's remaining queue record for a fully imported torrent.

        Sonarr closes a tracked download by itself only when ONE import covers the grab's full episode count.
        RETRY asks the manager for another try this run. FAILED keeps the pending import for a later run.
        """

        del pending
        return EffectStatus.SKIPPED

    @property
    @abstractmethod
    def supports_blocking_wait(self) -> bool:
        """Whether the arr run loop may run the blocking wait pass."""


class ArrStrategy[ItemT: ArrItem](ImportCompleter):
    """An Arr-specific strategy the arr run machinery drives: the Arr REST client plus its domain logic."""

    @abstractmethod
    def get_items(self) -> list[ItemT]:
        """Every Arr item to consider this arr run (also the arr run's start hook)."""

    @abstractmethod
    def filter_to_single(self, items: list[ItemT], item_id: int) -> list[ItemT]:
        """Narrow the item list to the single external id `item_id`."""

    @abstractmethod
    def item_mappings(
        self,
        item: ItemT,
        log_ignored: bool = True,
    ) -> dict[int, TitleMapping]:
        """Resolve the AniList ids mapped to one Arr item."""

    @property
    @abstractmethod
    def warms_episodes(self) -> bool:
        """Whether `prefetch_episodes` does real work. The arr run calls `prefetch_orders` only when it does."""

    @abstractmethod
    def prefetch_episodes(self, items: list[ItemT], *, progress: ProgressCallback | None = None) -> int:
        """Warm per-item network caches concurrently before the scan loop, returning how many were attempted."""

    @abstractmethod
    def prefetch_orders(self, items: list[ItemT], *, progress: ProgressCallback | None = None) -> int:
        """Refetch the due TVDB episode orders of the series the run reads, returning how many were written.

        Runs after `prefetch_episodes`, only when `warms_episodes`.
        """

    @abstractmethod
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Arr history records since `date`, or None on failure."""

    @abstractmethod
    def check_title(
        self,
        item: ItemT,
        al_id: int,
        mapping: TitleMapping,
    ) -> None:
        """Check one title for one Arr item: decide, grab or skip, cache the result."""

    @abstractmethod
    def pending_import_series_id(self, item: ItemT) -> int | None:
        """The Arr series id whose carried-over pending imports belong to this item.

        None turns off the arr run loop's per-item pending-import snapshot.
        """
