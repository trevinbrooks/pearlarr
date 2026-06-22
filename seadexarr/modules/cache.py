"""Persistent run cache: schema ownership, freshness checks, and writes.

``CacheStore`` owns the on-disk cache schema and all reads/writes against it:
the descriptor block (package version + config checksum), the per-arr
``anilist_entries`` records, and the freshness check that decides whether a
title needs re-processing. It mutates its in-memory ``data`` dict eagerly and
only persists at the run's save points, so a hard kill mid-run loses at most the
titles finished since the last save (they're simply re-checked next run, never
silently skipped).

Each arr instance constructs its own ``CacheStore`` that reads the file fresh —
a scheduled cycle runs Radarr (which saves ``cache.json``) then Sonarr (which
re-reads it), handing off through the *file*, not shared memory. Do not share a
single ``CacheStore`` across arrs.

Extracted from ``SeaDexArr`` in Phase 2 of the refactor (see
``REFACTOR_PLAN.md``); behaviour-preserving.
"""

import json
import os
from typing import Any

from seadex import EntryRecord

from .. import __version__

# Timestamp format for cache record fields (entry ``updated_at`` and the AniList
# meta ``fetched_at``). Lives here because the cache owns the record schema;
# consumers (the orchestrator and the Sonarr adapter) import it directly.
UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"


def save_json(
    data: dict,
    out_file: str,
    sort_cache: bool = False,
) -> None:
    """Save JSON prettily

    Args:
        data (dict): Data to be saved
        out_file (str): Path to JSON file
        sort_cache (bool, optional): Whether to sort cache files by AniList ID. Defaults to False.
    """

    if sort_cache:

        anilist_entries = data.get("anilist_entries")
        if anilist_entries is not None:
            for arr, arr_item in anilist_entries.items():
                keys = list(arr_item.keys())
                keys.sort(key=int)
                sorted_data = {key: arr_item[key] for key in keys}

                anilist_entries[arr] = sorted_data

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


class CacheStore:
    """Owns the cache file: schema, freshness checks, and persistence."""

    def __init__(self, path: str, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: str, *, config_checksum: str) -> "CacheStore":
        """Load the cache from disk (or create the schema) and reconcile it.

        Args:
            path (str): Path to the cache file.
            config_checksum (str): Current config-file checksum, stamped into the
                descriptor block so a changed config invalidates stale records.
        """

        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            store = cls(path, data)
            store._reconcile(config_checksum)
        else:
            # A freshly built schema already carries the current version and
            # checksum, so there is nothing to reconcile.
            store = cls(path, cls._initial_schema(config_checksum))

        return store

    @staticmethod
    def _initial_schema(config_checksum: str) -> dict[str, Any]:
        """Build a fresh cache: descriptor (version + checksum) + entry store."""

        return {
            "description": {
                "seadexarr_version": __version__,
                "config_checksum": config_checksum,
            },
            "anilist_entries": {},
        }

    def _reconcile(self, config_checksum: str) -> None:
        """Update the descriptor when the package version or config has changed."""

        if (
            self.data.get("description", {}).get("seadexarr_version", None)
            != __version__
        ):
            self.data["description"]["seadexarr_version"] = __version__

        if (
            self.data.get("description", {}).get("config_checksum", None)
            != config_checksum
        ):
            self.data["description"]["config_checksum"] = config_checksum

    def check_al_id_in_cache(
        self,
        arr: str,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """Check if timestamps in the cache match when SeaDex entry was last updated

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            seadex_entry: SeaDex entry
        """
        sd_time = seadex_entry.updated_at
        sd_time_str = sd_time.strftime(UPDATED_AT_STR_FORMAT)
        cache_time = (
            self.data.get("anilist_entries", {})
            .get(arr, {})
            .get(str(al_id), {})
            .get("updated_at")
        )

        return sd_time_str == cache_time

    def get_cached_name(
        self,
        arr: str,
        al_id: int,
    ) -> str | None:
        """Get the AniList title stored in the cache for an entry, if any

        The title is written into the cache alongside the timestamp when an
        entry is first processed, so it can be reused for cached entries
        without an additional AniList lookup.

        Args:
            arr (str): Arr instance the entry is cached under
            al_id (int): AniList ID

        Returns:
            str | None: Cached title, or None if not present
        """

        return self.get_cached_field(arr, al_id, "name")

    def get_cached_field(
        self,
        arr: str,
        al_id: int,
        field: str,
    ) -> Any:
        """Read a single stored field from an entry's cache record, if present

        Args:
            arr (str): Arr instance the entry is cached under
            al_id (int): AniList ID
            field (str): Cache field name (e.g. "name", "url", "coverage")

        Returns:
            The stored value, or None if absent
        """

        return (
            self.data.get("anilist_entries", {})
            .get(arr, {})
            .get(str(al_id), {})
            .get(field)
        )

    def update_cache(self, arr: str, al_id: int, cache_details: dict | None = None) -> bool:
        """Update cache with useful info

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            cache_details (dict): Details for the cache entry.
                Defaults to None
        """

        if cache_details is None:
            cache_details = {}

        if "updated_at" in cache_details:
            cache_details["updated_at"] = cache_details["updated_at"].strftime(
                UPDATED_AT_STR_FORMAT,
            )

        # Add to cache and save out
        if arr not in self.data["anilist_entries"]:
            self.data["anilist_entries"][arr] = {}

        if str(al_id) not in self.data["anilist_entries"][arr]:
            self.data["anilist_entries"][arr][str(al_id)] = {}

        self.data["anilist_entries"][arr][str(al_id)].update(cache_details)

        # Mutate the in-memory cache only - don't persist here. The run's save
        # points (the max_torrents_to_add early exits and the end-of-run save in
        # run()) flush it. This avoids re-serializing the whole cache - which
        # includes the large, mostly-static anilist_meta block - once per title,
        # turning N full-file writes per run into a handful.
        #
        # Trade-off: a hard kill mid-run loses the titles finished since the last
        # save point, so they're simply re-checked on the next run. That's the
        # safe direction - we never skip a title that wasn't durably recorded as
        # done. (A preview likewise only mutates in memory; the save below is
        # gated on the preview flag, so a preview still never persists.)

        return True

    def save(self, *, preview: bool, sort: bool = True) -> None:
        """Persist the in-memory cache to disk, unless this is a preview run.

        Skipped during a preview so a preview never writes state, mirroring
        ``update_cache`` only mutating memory.

        Args:
            preview (bool): When True, skip the write entirely.
            sort (bool): Sort anilist_entries by id before writing. Defaults to
                True so the persisted file is ordered by id; pass False to skip
                the sort on a hot write path.
        """

        if not preview:
            save_json(
                self.data,
                self.path,
                sort_cache=sort,
            )
