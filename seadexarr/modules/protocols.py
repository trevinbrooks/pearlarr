"""Composition seams between the run machinery and the Arr-specific strategies.

The refactor replaced the ``SeaDexArr`` ABC + subclass inheritance with
composition (see ``REFACTOR_PLAN.md``). Two Protocols express the seam:

- :class:`ArrSync` — what the run machinery calls on a strategy (the items to
  process and the per-id body). ``SonarrSync`` / ``RadarrSync`` satisfy it.
- :class:`RunServices` — the shared per-id pipeline a strategy invokes while
  processing one item. The composition root (``cli.py``) injects the
  ``SeaDexArr`` run machinery into each strategy, which holds it as this view and
  calls the pipeline through it (``SeaDexArr`` satisfies it structurally). The
  strategy depends on this abstraction, not the concrete engine.

Defining both here (and importing nothing from the engine or the strategies)
keeps the dependency graph acyclic: ``seadex_arr`` -> ``protocols`` and
``seadex_sonarr`` -> {``seadex_arr``, ``protocols``}, with no import cycle.
"""

from collections.abc import Callable
from typing import Any, Protocol

from seadex import EntryRecord


class RunServices(Protocol):
    """Engine operations a strategy invokes while processing one item.

    This is the contract the strategy depends on; ``SeaDexArr`` implements it.
    Mutations of run state (the ``RunContext``) happen behind these calls, on
    the engine, so the strategy never touches that state directly.
    """

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: str = "movie",
        log_ignored: bool = True,
    ) -> dict: ...

    def al_id_prologue(self, al_id: int | None) -> EntryRecord | None: ...

    def cached_entry_skip(
        self,
        arr: str,
        al_id: int,
        sd_entry: EntryRecord,
        sd_url: str,
        coverage: Callable[[], str],
    ) -> bool: ...

    def check_al_id_in_cache(
        self,
        arr: str,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool: ...

    def get_anilist_title(self, al_id: int) -> str: ...

    def get_seadex_dict(self, sd_entry: EntryRecord) -> dict: ...

    def filter_seadex_interactive(
        self,
        seadex_dict: dict,
        sd_entry: EntryRecord,
    ) -> dict: ...

    def filter_seadex_downloads(
        self,
        al_id: int,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        ep_list: list | None = None,
    ) -> tuple[list, dict]: ...

    def grab_and_cache(
        self,
        arr: str,
        al_id: int,
        item_title: str,
        anilist_title: str,
        sd_url: str,
        seadex_dict: dict,
        torrent_hashes: list,
        cache_details: dict,
        release_group: list | str | None,
    ) -> bool: ...

    def update_cache(
        self,
        arr: str,
        al_id: int,
        cache_details: dict | None = None,
    ) -> bool: ...

    def log_no_seadex_releases(self) -> bool: ...

    def log_entry_status(
        self,
        state: str,
        label: str,
        style: str | None = "grey50",
    ) -> bool: ...

    def log_cached_entry(
        self,
        arr: str,
        al_id: int,
        state: str = "unchanged",
    ) -> bool: ...

    def log_anilist_item_unmonitored(self, item_title: str) -> bool: ...

    def log_al_title(
        self,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> bool: ...


class ArrSync(Protocol):
    """An Arr-specific sync strategy the run machinery drives.

    Owns the Arr REST client and the Arr's domain logic (episode mapping,
    release-group resolution). Provides the items to process and the per-id
    body. The strategy is injected with its :class:`RunServices` and holds it,
    so the run loop calls these hooks without passing itself.
    """

    def get_items(self) -> list: ...

    def filter_to_single(self, items: list, item_id: int) -> list: ...

    def item_anilist_ids(
        self,
        item: Any,
        log_ignored: bool = True,
    ) -> dict: ...

    def process_al_id(
        self,
        arr: str,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: dict,
    ) -> bool: ...
