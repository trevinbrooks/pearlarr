from abc import ABC, abstractmethod
from typing import Any

from .config import Arr
from .mappings import MappingEntry


class ArrSync(ABC):
    """An Arr-specific sync strategy the run machinery drives.

    Owns the Arr REST client and the Arr's domain logic (episode mapping,
    release-group resolution). Provides the items to process and the per-id
    body. The strategy is injected with the :class:`~.seadex_arr.SeaDexArr` run
    machinery and holds it, so the run loop calls these hooks without passing
    itself. Subclasses (``SonarrSync`` / ``RadarrSync``) must implement every
    hook; the ABC enforces that at instantiation.
    """

    @abstractmethod
    def get_items(self) -> list:
        """Every Arr item to consider this run (also the run-start hook)."""

    @abstractmethod
    def filter_to_single(self, items: list, item_id: int) -> list:
        """Narrow the item list to the single external id ``item_id``."""

    @abstractmethod
    def item_anilist_ids(
        self,
        item: Any,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve the AniList ids mapped to one Arr item."""

    @abstractmethod
    def process_al_id(
        self,
        arr: Arr,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for one Arr item; True if it grabbed."""
