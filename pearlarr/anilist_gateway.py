"""AniList client gateway: the in-memory meta cache, its refresh age, and prefetch.

`AniListGateway` owns the per-run `al_cache` (AniList responses keyed by id)
and the persisted `anilist_meta` block in the cache file: it seeds the cache
from disk, batch-fetches whatever is missing or past the refresh age, persists
what it fetched, and resolves titles / thumbnails. A record past the refresh
age still serves, so an AniList outage never drains the cache.

The gateway is deliberately side-effect-free with respect to run state - the
caller owns the `current_title` attribution.
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from .anilist_client import ANILIST_BATCH_SIZE, AniListCache, AniListClient, extract_path, media_from, media_node_from
from .cache import UPDATED_AT_STR_FORMAT, AbstractCacheStore, record_payload, stamp_is_fresh
from .log import count_noun
from .seadex_types import AniListMediaNode, ProgressSink

# How old a persisted AniList response gets before a run refetches it.
# title/format/coverImage are effectively static. Episodes for a currently airing
# show drift, so this caps how stale that count can get (~one episode/week).
ANILIST_CACHE_TTL_DAYS = 7


class AniListGateway:
    """In-memory AniList meta cache backed by the persisted cache file."""

    def __init__(
        self,
        *,
        cache_store: AbstractCacheStore,
        logger: logging.Logger,
        client: AniListClient,
    ) -> None:
        """Wire the gateway to the cache store, logger, and the bound wire client."""

        self._cache = cache_store
        self.logger = logger
        self._client = client
        self.al_cache: AniListCache = {}
        # Loaded past the refresh age: served, refetched, and the episode count held back.
        self._stale: set[int] = set()
        # Fetched this run: exactly what the save writes.
        self._fetched: set[int] = set()
        # Unknown to AniList this run: no accessor re-queries them.
        self._absent: set[int] = set()

    @property
    def outage(self) -> bool:
        """True once AniList has been declared unavailable for this run."""

        return self._client.outage

    def _store(self, al_id: int, body: dict[str, Any]) -> None:
        """Put a freshly fetched body in the run cache and queue it for the save."""

        self.al_cache[al_id] = body
        self._stale.discard(al_id)
        self._fetched.add(al_id)

    def load_cache(self) -> None:
        """Seed the run cache from every stored record, marking those past the refresh age stale."""

        cutoff = datetime.now() - timedelta(days=ANILIST_CACHE_TTL_DAYS)
        loaded = 0
        for al_id, record in self._cache.iter_anilist_meta():
            payload = record_payload(record, "data")
            if payload is None:
                continue
            self.al_cache[al_id] = payload
            # An aged or unreadable stamp still serves: refetch, never drop.
            if not stamp_is_fresh(record, cutoff):
                self._stale.add(al_id)
            loaded += 1

        if loaded:
            self.logger.debug(f"Loaded {count_noun(loaded, 'AniList entry', 'AniList entries')} from cache")

    def save_cache(self, *, preview: bool) -> None:
        """Persist this run's fetches stamped now. The prefetch-time save is the only save.

        Rows past the refresh age are evicted only on a healthy, non-preview run:
        a stale id the batch did not return is gone, and absent next run.
        """

        now = datetime.now()
        now_str = now.strftime(UPDATED_AT_STR_FORMAT)
        written = len(self._fetched)
        for al_id in sorted(self._fetched):
            self._cache.put_anilist_meta(al_id, {"fetched_at": now_str, "data": self.al_cache[al_id]})
        self._fetched.clear()

        # Evicting during an outage would drain the very records serving the run.
        cutoff = now - timedelta(days=ANILIST_CACHE_TTL_DAYS)
        evicted = 0 if preview or self._client.outage else self._cache.evict_anilist_meta(cutoff)

        if written or evicted:
            self._cache.save(preview=preview)
        if evicted:
            self.logger.debug(f"Evicted {count_noun(evicted, 'stale AniList meta record')}")

    def prefetch(
        self,
        al_ids: Iterable[int],
        *,
        preview: bool,
        progress: ProgressSink | None = None,
    ) -> int:
        """Batch-fetch the missing and stale ids, then persist. Returns how many needed fetching.

        `progress` is the boot cockpit step fed per-batch fraction + "done/total".
        """

        wanted = sorted({i for i in al_ids if i not in self.al_cache or i in self._stale})
        total = len(wanted)
        if not total:
            return 0

        done = 0
        for start in range(0, total, ANILIST_BATCH_SIZE):
            # A tripped breaker answers every later batch empty, so stop asking.
            if self._client.outage:
                break
            chunk = wanted[start : start + ANILIST_BATCH_SIZE]
            fetched = self._client.query_batch(chunk)
            for al_id, body in fetched.items():
                self._store(al_id, body)
            # A healthy batch answers for its whole chunk: what it left out is unknown to AniList.
            if not self._client.outage:
                self._absent.update(i for i in chunk if i not in fetched)
            done += len(chunk)
            if progress is not None:
                progress.progress(done / total, f"{done}/{total}")

        # Persist before the main loop so the batch's work survives an early return.
        self.save_cache(preview=preview)
        return total

    def _media(self, al_id: int) -> AniListMediaNode:
        """Resolve the typed Media node for an id: the run cache, then one wire query. All-None on a miss."""

        body = self.al_cache.get(al_id)
        if body is not None:
            return media_from(body)
        # A remembered miss or a tripped breaker answers without a request.
        if al_id in self._absent or self._client.outage:
            return AniListMediaNode()

        fetched = self._client.query(al_id)
        raw_media = extract_path(fetched, "data", "Media")
        if raw_media:
            self._store(al_id, fetched)
        elif not self._client.outage:
            # AniList answered and knows no such id. A failure trips the breaker instead.
            self._absent.add(al_id)
        return media_node_from(raw_media)

    def title(self, al_id: int) -> str | None:
        """Resolve the AniList title for an id (cache or live query), or None.

        Prefers the English title, falling back to romaji. Side-effect-free: the
        caller owns any fallback and the `current_title` attribution.
        """

        media = self._media(al_id)
        return media.title_english or media.title_romaji

    def thumb(self, al_id: int) -> str | None:
        """Resolve the AniList cover thumbnail URL for an id, or None."""

        return self._media(al_id).cover_image

    def banner(self, al_id: int) -> str | None:
        """Resolve the AniList wide banner URL for an id, or None."""

        return self._media(al_id).banner_image

    def media_format(self, al_id: int) -> str | None:
        """Resolve the AniList media format (TV / MOVIE / OVA / ...) for an id, or None."""

        return self._media(al_id).format

    def n_eps(self, al_id: int) -> int | None:
        """Resolve the AniList episode count for an id, or None (also while its record is stale)."""

        # The count is the one field the refresh age exists for. None downstream means "use every episode".
        if al_id in self._stale:
            return None
        return self._media(al_id).episodes
