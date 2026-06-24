from abc import ABC, abstractmethod

from .config import Arr
from .mappings import MappingEntry
from .seadex_types import ArrItem


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
    def process_al_id(
        self,
        arr: Arr,
        item: ItemT,
        item_title: str,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for one Arr item; True if it grabbed."""
