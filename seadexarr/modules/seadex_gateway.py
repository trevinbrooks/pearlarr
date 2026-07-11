"""SeaDex API gateway: bulk-prefetch entries, then serve lookups by AniList id.

``SeaDexGateway`` wraps the SeaDex client with the connection-error handling the
orchestrator relies on: a missing entry and a SeaDex outage both degrade to a
typed :class:`SeaDexMiss` rather than raising, and the two are distinguishable
so an outage skip is never reported as "no entry".

The hot path used to be one ``from_id`` round-trip per library id - hundreds per
run, just to read each entry's ``updated_at`` and usually skip. :meth:`prefetch`
collapses that into ``ceil(N / SEADEX_BATCH_SIZE)`` batched ``from_filter`` queries
(OR-ed ``alID`` clauses), mirroring the AniList prefetch, and :meth:`entry` then
serves from the warmed per-run cache. The gateway is rebuilt per arr run, so the
cache never crosses runs (entries are stable within a run, may change between).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from itertools import batched
from typing import Protocol, override

import httpx
from seadex import EntryNotFoundError, EntryRecord

from .output import Severity, hub_note
from .seadex_types import ProgressSink

# Ids per batched ``from_filter`` query. Mirrors ANILIST_BATCH_SIZE; ~50
# ``alID=NNNNNN`` clauses stay well under the GET URL length limit (the spike
# confirmed the OR-filter form and that perPage=500 means a 50-id batch never
# paginates).
SEADEX_BATCH_SIZE = 50


class SeaDexMiss(Enum):
    """Why :meth:`SeaDexSource.entry` has no record for an id.

    NO_ENTRY is SeaDex's answer (the id genuinely has no entry); OUTAGE means the
    lookup was skipped because SeaDex is unreachable this run - the caller must
    report those distinctly (an outage skip is not a missing entry).
    """

    NO_ENTRY = auto()
    OUTAGE = auto()


class SeaDexSource(ABC):
    """The SeaDex lookup surface the run engine consumes.

    A nominal seam over what the engine reads off ``deps.seadex``
    (:meth:`prefetch` to warm the per-run cache, :meth:`entry` to serve a lookup,
    :attr:`outage` for the run's SeaDex-unreachable state). The real
    :class:`SeaDexGateway` subclasses it, so a test can inject a typed,
    network-free stand-in via ``RunDeps.seadex`` that's checked against this
    surface instead of laundered through a bare ``object.__new__`` instance.
    """

    @abstractmethod
    def prefetch(self, al_ids: Iterable[int], *, progress: ProgressSink | None = None) -> int: ...

    @abstractmethod
    def entry(self, al_id: int) -> EntryRecord | SeaDexMiss: ...

    @property
    @abstractmethod
    def outage(self) -> bool: ...


class SeaDexEntryApi(Protocol):
    """The slice of the ``seadex`` lib's ``SeaDexEntry`` client the gateway consumes.

    Structural, so tests inject a network-free stand-in through the real
    constructor instead of bypassing it. Positional-only params mirror the lib's
    actual signatures (its ``from_id`` accepts ``int | str``, wider is fine).
    """

    def from_filter(self, filter_str: str, /) -> Iterable[EntryRecord]: ...

    def from_id(self, al_id: int, /) -> EntryRecord: ...


class SeaDexGateway(SeaDexSource):
    """Thin wrapper over the SeaDex client: bulk prefetch + by-id lookups."""

    def __init__(self, *, client: SeaDexEntryApi) -> None:
        """Wire the gateway to the SeaDex client it decorates.

        Args:
            client (SeaDexEntryApi): The SeaDex API client every lookup rides.
        """

        self._client = client
        # Per-run bulk-fetch cache: entries keyed by AniList id, plus the set of ids
        # we successfully prefetched - so an id that was requested but not returned
        # is known-absent and ``entry`` can skip the per-id fallback call.
        self._entry_cache: dict[int, EntryRecord] = {}
        self._prefetched: set[int] = set()
        # Set once a request has failed twice (batch and single-id lookups both
        # get one immediate retry): SeaDex is down for this run, so every later
        # batch/lookup short-circuits instead of re-timing-out per id.
        self._outage = False

    @property
    @override
    def outage(self) -> bool:
        """True once SeaDex has been declared unreachable for this run."""

        return self._outage

    def _note_outage(self, e: httpx.HTTPError) -> None:
        """Warn ONCE that SeaDex is unreachable; the flag mutes every later call."""

        if not self._outage:
            hub_note(
                f"SeaDex request failed ({type(e).__name__}); affected titles will be skipped this run",
                severity=Severity.WARNING,
            )
        self._outage = True

    @override
    def prefetch(self, al_ids: Iterable[int], *, progress: ProgressSink | None = None) -> int:
        """Bulk-fetch SeaDex entries for many ids in batched OR-filter queries.

        Populates the per-run entry cache so the per-item loop's :meth:`entry`
        calls are cache hits instead of one ``from_id`` round-trip each. Ids in a
        successfully-fetched batch that SeaDex didn't return are remembered as
        absent (so :meth:`entry` reports them NO_ENTRY without a call). A failed
        batch gets ONE immediate retry (a single transient 502 must not write off
        the whole run); a batch that fails twice warns once and flips the outage
        flag, so the remaining batches - and every later :meth:`entry` fallback -
        short-circuit to an OUTAGE miss instead of re-timing-out per id.

        Args:
            al_ids (Iterable[int]): Candidate AniList IDs for this run.
            progress (ProgressSink | None): Boot cockpit step fed per-batch
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
        for chunk in batched(missing, SEADEX_BATCH_SIZE, strict=False):
            chunk = list(chunk)
            # A failed chunk's ids are NOT marked prefetched (never "absent"), so a
            # transient miss stays a skip, not a remembered no-entry.
            if not self._outage:
                fetched = self._fetch_chunk(chunk)
                if fetched is not None:
                    self._entry_cache.update(fetched)
                    self._prefetched.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress.progress(done / total, f"{done}/{total}")

        return total

    def _fetch_chunk(self, chunk: list[int]) -> dict[int, EntryRecord] | None:
        """One prefetch batch with a single immediate retry.

        A lone transient blip (one 502 among many batches) is absorbed silently;
        only a chunk that fails twice declares the outage and returns None.
        """

        try:
            return self._fetch_batch(chunk)
        except httpx.HTTPError:
            pass
        try:
            return self._fetch_batch(chunk)
        except httpx.HTTPError as e:
            self._note_outage(e)
            return None

    def _fetch_batch(self, al_ids: list[int]) -> dict[int, EntryRecord]:
        """Fetch one batch via an OR-ed ``alID`` filter, keyed by AniList id."""

        filter_str = " || ".join(f"alID={al_id}" for al_id in al_ids)
        return {record.anilist_id: record for record in self._client.from_filter(filter_str)}

    @override
    def entry(self, al_id: int) -> EntryRecord | SeaDexMiss:
        """Get the SeaDex entry for an AniList id, or a typed miss.

        Served from the bulk-prefetch cache when warm; a prefetched-but-absent id
        is NO_ENTRY without a call; an id that wasn't prefetched (e.g. a single-id
        run) falls back to a single ``from_id`` - unless the outage flag is set, in
        which case it degrades straight to OUTAGE without another network attempt.

        Args:
            al_id (int): AniList ID.
        """

        cached = self._entry_cache.get(al_id)
        if cached is not None:
            return cached
        if al_id in self._prefetched:
            return SeaDexMiss.NO_ENTRY
        if self._outage:
            return SeaDexMiss.OUTAGE
        return self._entry_single(al_id)

    def _entry_single(self, al_id: int) -> EntryRecord | SeaDexMiss:
        """Single by-id lookup with the orchestrator's graceful degradation.

        A missing entry (``EntryNotFoundError``) is NO_ENTRY, immediately - that's
        SeaDex answering, not failing. A failed request (any ``httpx.HTTPError`` -
        connection failure, timeout, HTTP error status; the same breadth
        :meth:`prefetch` catches) gets ONE immediate silent retry, mirroring
        :meth:`_fetch_chunk` (a single transient 502 must not write off the whole
        run); only a lookup that fails twice is OUTAGE, so the caller can report
        the skip truthfully - it warns once and mutes later lookups.
        """

        try:
            return self._client.from_id(al_id)
        except EntryNotFoundError:
            return SeaDexMiss.NO_ENTRY
        except httpx.HTTPError:
            pass
        try:
            return self._client.from_id(al_id)
        except EntryNotFoundError:
            return SeaDexMiss.NO_ENTRY
        except httpx.HTTPError as e:
            self._note_outage(e)
            return SeaDexMiss.OUTAGE
