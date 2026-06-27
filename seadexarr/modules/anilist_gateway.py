"""AniList client gateway: the in-memory meta cache, its TTL, and prefetch.

``AniListGateway`` owns the per-run ``al_cache`` (AniList responses keyed by id)
and the persisted ``anilist_meta`` block in the cache file: it seeds the cache
from disk, batch-fetches everything still missing, persists newly seen
responses (respecting a TTL), and resolves titles / thumbnails.

Extracted from ``SeaDexArr`` during the refactor; behaviour-preserving. The
gateway is deliberately
side-effect-free with respect to the orchestrator's run state - ``title`` no
longer stamps ``current_title``; the caller does that.
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta

from .anilist import (
    ANILIST_BATCH_SIZE,
    AniListCache,
    get_anilist_thumb,
    get_anilist_title,
    get_query_batch,
)
from .cache import UPDATED_AT_STR_FORMAT, CacheStore, record_is_fresh
from .log import indent_string

# How long a persisted AniList response stays usable before it's re-fetched.
# title/format/coverImage are effectively static; episodes for a currently airing
# show drift, so this caps how stale that count can get (~one episode/week).
ANILIST_CACHE_TTL_DAYS = 7


class AniListGateway:
    """In-memory AniList meta cache backed by the persisted cache file."""

    def __init__(
        self,
        *,
        cache_store: CacheStore,
        logger: logging.Logger,
    ) -> None:
        """Wire the gateway to the cache store and logger.

        Args:
            cache_store (CacheStore): Owns the on-disk ``anilist_meta`` block and
                the preview-gated save.
            logger (logging.Logger): For the prefetch / load progress lines.
        """

        self._cache = cache_store
        self.logger = logger
        self.al_cache: AniListCache = {}

    def load_cache(self) -> None:
        """Seed the in-memory AniList cache from the persisted store

        AniList metadata (title / format / episodes / cover) is effectively
        static, so reusing what we fetched on previous runs is what keeps a run
        from re-querying AniList for ids it has already seen - the main cause of
        the rate-limit stalls. Entries older than ANILIST_CACHE_TTL_DAYS are
         skipped, so the data can't get arbitrarily stale (see prefetch /
        save_cache for the writing side).
        """

        cutoff = datetime.now() - timedelta(days=ANILIST_CACHE_TTL_DAYS)
        loaded = 0
        for al_id, record in self._cache.iter_anilist_meta():
            if not record_is_fresh(
                record,
                payload_key="data",
                ttl_days=ANILIST_CACHE_TTL_DAYS,
                cutoff=cutoff,
            ):
                continue
            self.al_cache[al_id] = record["data"]
            loaded += 1

        if loaded:
            self.logger.debug(
                indent_string(f"Loaded {loaded} AniList entries from cache"),
            )

    def save_cache(self, *, preview: bool) -> None:
        """Persist any newly seen AniList responses back to the on-disk cache

        An entry that's already stored and still fresh keeps its original
        fetched_at (so the TTL actually expires it rather than resetting every
        run); a missing OR stale entry is (re)written with the current time, so
        an aged-out id is refreshed instead of being re-fetched on every run.

        Args:
            preview (bool): When True, keep the warmed entries in memory but
                don't persist them (the gate lives in ``CacheStore.save``).
        """

        now = datetime.now()
        now_str = now.strftime(UPDATED_AT_STR_FORMAT)
        cutoff = now - timedelta(days=ANILIST_CACHE_TTL_DAYS)

        written = 0
        for al_id, data in self.al_cache.items():
            if record_is_fresh(
                self._cache.get_anilist_meta(al_id),
                payload_key="data",
                ttl_days=ANILIST_CACHE_TTL_DAYS,
                cutoff=cutoff,
            ):
                continue
            self._cache.put_anilist_meta(al_id, {"fetched_at": now_str, "data": data})
            written += 1

        # Drop meta records aged past the same TTL we refuse to read, so the block
        # stops accumulating dead weight (the un-evicted-stale problem). Skipped in
        # preview (which persists nothing anyway).
        evicted = 0 if preview else self._cache.evict_anilist_meta(cutoff)

        if written or evicted:
            self._cache.save(preview=preview)
        if evicted:
            self.logger.debug(
                indent_string(f"Evicted {evicted} stale AniList meta record(s)"),
            )

    def prefetch(self, al_ids: Iterable[int], *, preview: bool) -> None:
        """Warm the AniList cache for a set of ids in batched requests

        Fetches everything still missing from the cache in ANILIST_BATCH_SIZE-id
        "id_in" pages (one request per page) instead of one request per id on
        demand, then persists the results. This is what collapses a cold run's
        ~one-AniList-request-per-series into a handful, so the per-title loop
        rarely has to hit AniList one id at a time and trip its rate limit.

        Args:
            al_ids (iterable[int]): Candidate AniList IDs for this run
            preview (bool): Forwarded to the post-fetch save's preview gate.
        """

        missing = sorted(
            {i for i in al_ids if i not in self.al_cache},
        )
        if not missing:
            return

        # Surfaced at INFO (only when there's actually something to fetch, so
        # warm runs stay silent), so the upfront pause on a cold run is explained
        self.logger.info(
            indent_string(
                f"Prefetching {len(missing)} AniList entries "
                f"in batches of {ANILIST_BATCH_SIZE}",
            ),
            extra={"line_style": "grey50"},
        )

        for start in range(0, len(missing), ANILIST_BATCH_SIZE):
            chunk = missing[start:start + ANILIST_BATCH_SIZE]
            # Ids unknown to AniList are simply absent from the result; the
            # per-id helpers will try once more on demand and degrade gracefully
            for al_id, data in get_query_batch(chunk).items():
                self.al_cache[al_id] = data

        # Persist now (before the main loop) so the batch's work survives even an
        # early return - e.g., when max_torrents_to_add is hit mid-run
        self.save_cache(preview=preview)

    def title(self, al_id: int) -> str | None:
        """Resolve the AniList title for an id (cache or live query), or None.

        Side-effect-free: the caller owns any fallback and the ``current_title``
        attribution.

        Args:
            al_id (int): AniList ID
        """

        anilist_title, self.al_cache = get_anilist_title(
            al_id,
            al_cache=self.al_cache,
        )
        return anilist_title

    def thumb(self, al_id: int) -> str | None:
        """Resolve the AniList cover thumbnail URL for an id, or None.

        Args:
            al_id (int): AniList ID
        """

        anilist_thumb, self.al_cache = get_anilist_thumb(
            al_id,
            al_cache=self.al_cache,
        )
        return anilist_thumb
