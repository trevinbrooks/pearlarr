from abc import ABC, abstractmethod
from typing import Protocol

from .config import Arr
from .manual_import import ImportProbe, PendingImport
from .mappings import MappingEntry
from .seadex_types import ArrItem


class ImportCompleter(Protocol):
    """The single strategy hook the engine drives after a download completes.

    Narrow and NON-generic, so the engine can hold the active strategy as
    ``ImportCompleter | None`` and call exactly this one method without the
    invariant-``ArrSync[ItemT]`` cast (the engine never touches ``ItemT``).
    ``ArrSync`` structurally satisfies it, so a concrete strategy assigns to an
    ``ImportCompleter`` slot with no cast.
    """

    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe: ...


class ArrSync[ItemT: ArrItem](ABC):
    """An Arr-specific sync strategy the run machinery drives.

    Owns the Arr REST client and the Arr's domain logic (episode mapping,
    release-group resolution). Provides the items to process and the per-id
    body. The strategy is injected with the :class:`~.seadex_arr.SeaDexArr` run
    machinery and holds it, so the run loop calls these hooks without passing
    itself. Subclasses (``SonarrSync`` / ``RadarrSync``) must implement every
    hook; the ABC enforces that at instantiation.

    Generic in ``ItemT`` (the Arr's item protocol — :class:`~.seadex_types.SonarrItem`
    or :class:`~.seadex_types.RadarrItem`) so each subclass binds its own item
    type without the loose ``list``/``Any`` the base used to carry. ``ArrSync`` is
    invariant in ``ItemT`` (it appears in both inputs and outputs), so a concrete
    strategy must reach the generic ``run_sync[ItemT]``: the composition root
    branches per Arr to bind one item type per call, and the run loop only ever
    touches the shared ``ArrItem`` surface (``.monitored``/``.title``) off items.
    """

    @abstractmethod
    def get_items(self) -> list[ItemT]:
        """Every Arr item to consider this run (also the run-start hook)."""

    @abstractmethod
    def filter_to_single(self, items: list[ItemT], item_id: int) -> list[ItemT]:
        """Narrow the item list to the single external id ``item_id``."""

    @abstractmethod
    def item_anilist_ids(
        self,
        item: ItemT,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve the AniList ids mapped to one Arr item."""

    @abstractmethod
    def prefetch_episodes(self, items: list[ItemT]) -> None:
        """Warm per-item network caches concurrently before the scan loop.

        Called once in the pre-scan prefetch step, beside the AniList/SeaDex bulk
        prefetches. ``SonarrSync`` fans the per-series ``/api/v3/episode`` fetches
        out over a bounded pool; ``RadarrSync`` is a no-op (no episodes).
        """

    @abstractmethod
    def process_al_id(
        self,
        arr: Arr,
        item: ItemT,
        item_title: str,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for one Arr item; True if it grabbed."""

    @abstractmethod
    def pending_import_series_id(self, item: ItemT) -> int | None:
        """The Arr series id whose carried-over pending records this item owns.

        The key for the engine's per-item non-blocking snapshot hook: after all
        of an item's AniList ids are processed, the engine reconciles+reports any
        carried-over pending records for this series id inline. Sonarr returns
        ``item.id``; Radarr returns ``None`` (movies record no pending imports),
        which short-circuits the snapshot entirely.
        """

    @abstractmethod
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Reconcile one completed download with Sonarr (one poll).

        Called repeatedly by the engine's monitor/snapshot once qBittorrent
        reports the torrent complete. Reads Sonarr's (refreshed) queue and the
        current episode files as the source of truth: lets Sonarr finish when it
        is actively importing, treats target episodes that already hold the
        recommended release as imported, and otherwise drives a series-pinned
        manual import using *our* authoritative file->episode mapping (never
        Sonarr's blind parse, so it can't import an episode our mapping assigned
        to another preferred torrent). Radarr is a no-op (out of scope).

        Args:
            pending (PendingImport): The durable record for the completed torrent.
            content_path (str): The qBittorrent ``content_path`` of the finished
                download (the folder/file the manual import reads from disk).
            force (bool): When True, stop deferring to Sonarr on a clean
                ``importPending`` and drive our manual import now. The engine sets
                this on the snapshot/reconcile passes and on the final in-bound
                monitor poll, so a download Sonarr will never import (e.g.
                Completed Download Handling off) is still imported rather than
                waited on forever.
            at_deadline (bool): When True, this is the final attempt for the
                record, so an intended file still not visible is terminal -> warn
                loudly. Off the deadline a still-missing file is expected (an
                early poll) and only logged at debug. Distinct from ``force``: the
                snapshot/reconcile force without being at a deadline (no warning).

        Returns:
            ImportProbe: the readiness (drop / retry / leave) plus whether the
            intended episode files are verified present (``files_present``) and
            whether an import command was just accepted (``command_issued``).
        """
