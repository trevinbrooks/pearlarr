"""SeaDex API gateway: bulk-prefetch entries, then serve lookups by AniList id.

``SeaDexGateway`` wraps the SeaDex client with the connection-error handling the
orchestrator relies on (a missing entry and a SeaDex outage both degrade to
``None`` rather than raising).

The hot path used to be one ``from_id`` round-trip per library id - hundreds per
run, just to read each entry's ``updated_at`` and usually skip. :meth:`prefetch`
collapses that into ``ceil(N / SEADEX_BATCH_SIZE)`` batched ``from_filter`` queries
(OR-ed ``alID`` clauses), mirroring the AniList prefetch, and :meth:`entry` then
serves from the warmed per-run cache. The gateway is rebuilt per arr run, so the
cache never crosses runs (entries are stable within a run, may change between).
"""

import logging
from collections.abc import Iterable
from itertools import batched
from typing import Protocol

import httpx
from seadex import EntryNotFoundError, EntryRecord, SeaDexEntry

# Ids per batched ``from_filter`` query. Mirrors ANILIST_BATCH_SIZE; ~50
# ``alID=NNNNNN`` clauses stay well under the GET URL length limit (the spike
# confirmed the OR-filter form and that perPage=500 means a 50-id batch never
# paginates).
SEADEX_BATCH_SIZE = 50


class PrefetchProgress(Protocol):
    """Sink for batch-prefetch progress - drives the boot cockpit's live bar.

    Structural, so the boot view's step handle satisfies it without this gateway
    importing the UI layer (mirrors ``mappings.DownloadProgress``).
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...


class SeaDexGateway:
    """Thin wrapper over the SeaDex client: bulk prefetch + by-id lookups."""

    def __init__(self, *, logger: logging.Logger) -> None:
        """Instantiate the SeaDex API client.

        Args:
            logger (logging.Logger): For the prefetch notice and the
                connection-error warning.
        """

        self.logger = logger
        self.seadex = SeaDexEntry()
        # Per-run bulk-fetch cache: entries keyed by AniList id, plus the set of ids
        # we successfully prefetched - so an id that was requested but not returned
        # is known-absent and ``entry`` can skip the per-id fallback call.
        self._entry_cache: dict[int, EntryRecord] = {}
        self._prefetched: set[int] = set()

    def prefetch(self, al_ids: Iterable[int], *, progress: PrefetchProgress | None = None) -> int:
        """Bulk-fetch SeaDex entries for many ids in batched OR-filter queries.

        Populates the per-run entry cache so the per-item loop's :meth:`entry`
        calls are cache hits instead of one ``from_id`` round-trip each. Ids in a
        successfully-fetched batch that SeaDex didn't return are remembered as
        absent (so :meth:`entry` returns None for them without a call). A batch that
        fails (outage) is left out of the prefetched set, so those ids fall through
        to the per-id fallback (which warns and degrades to None) on demand.

        Args:
            al_ids (Iterable[int]): Candidate AniList IDs for this run.
            progress (PrefetchProgress | None): Boot cockpit step fed per-batch
                fraction + "done/total" detail; None outside the cockpit.

        Returns:
            int: How many ids needed fetching (0 = nothing to do), for the
            caller's ledger detail.
        """

        missing = sorted({i for i in al_ids if i not in self._entry_cache and i not in self._prefetched})
        total = len(missing)
        if not total:
            return 0

        done = 0
        for chunk in batched(missing, SEADEX_BATCH_SIZE):
            chunk = list(chunk)
            try:
                fetched = self._fetch_batch(chunk)
            except httpx.HTTPError:
                # Leave this chunk for the per-id fallback rather than treating its
                # ids as absent; the fallback warns + degrades to None.
                self.logger.warning("Could not connect to SeaDex. Website may be down")
            else:
                self._entry_cache.update(fetched)
                self._prefetched.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress.progress(done / total, f"{done}/{total}")

        return total

    def _fetch_batch(self, al_ids: list[int]) -> dict[int, EntryRecord]:
        """Fetch one batch via an OR-ed ``alID`` filter, keyed by AniList id."""

        filter_str = " || ".join(f"alID={al_id}" for al_id in al_ids)
        return {record.anilist_id: record for record in self.seadex.from_filter(filter_str)}

    def entry(self, al_id: int) -> EntryRecord | None:
        """Get the SeaDex entry for an AniList id, or None.

        Served from the bulk-prefetch cache when warm; a prefetched-but-absent id
        returns None without a call; an id that wasn't prefetched (e.g. a single-id
        run) falls back to a single ``from_id``.

        Args:
            al_id (int): AniList ID.
        """

        cached = self._entry_cache.get(al_id)
        if cached is not None:
            return cached
        if al_id in self._prefetched:
            return None
        return self._entry_single(al_id)

    def _entry_single(self, al_id: int) -> EntryRecord | None:
        """Single by-id lookup with the orchestrator's graceful degradation.

        A missing entry (``EntryNotFoundError``) and a SeaDex outage
        (``httpx.ConnectError``) both return None so the caller can skip the id; the
        outage is surfaced as a warning.
        """

        sd_entry = None
        try:
            sd_entry = self.seadex.from_id(al_id)
        except EntryNotFoundError:
            pass
        except httpx.ConnectError:
            self.logger.warning("Could not connect to SeaDex. Website may be down")

        return sd_entry
