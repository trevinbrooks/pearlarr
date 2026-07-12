"""AniList client gateway: the in-memory meta cache, its TTL, and prefetch.

`AniListGateway` owns the per-run `al_cache` (AniList responses keyed by id)
and the persisted `anilist_meta` block in the cache file: it seeds the cache
from disk, batch-fetches everything still missing, persists newly seen
responses (respecting a TTL), and resolves titles / thumbnails.

Extracted from `RunLoop` during the refactor; behavior-preserving. The
gateway is deliberately
side-effect-free with respect to the orchestrator's run state - `title` no
longer stamps `current_title`; the caller does that.
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta

from .anilist_client import ANILIST_BATCH_SIZE, AniListCache, AniListClient, extract_path, media_from, media_node_from
from .cache import UPDATED_AT_STR_FORMAT, AbstractCacheStore, record_is_fresh
from .log import count_noun
from .seadex_types import AniListMediaNode, ProgressSink

# How long a persisted AniList response stays usable before it's re-fetched.
# title/format/coverImage are effectively static; episodes for a currently airing
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
        """Wire the gateway to the cache store, logger and wire client.

        Args:
            cache_store: Owns the on-disk `anilist_meta`
                block and the preview-gated save.
            logger: For the prefetch / load progress lines.
            client: The bound AniList wire client every lookup
                rides (it carries the shared web client and the per-run retry
                narration).
        """

        self._cache = cache_store
        self.logger = logger
        self._client = client
        self.al_cache: AniListCache = {}

    def load_cache(self) -> None:
        """Seed the in-memory AniList cache from the persisted store.

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
                cutoff=cutoff,
            ):
                continue
            self.al_cache[al_id] = record["data"]
            loaded += 1

        if loaded:
            self.logger.debug(f"Loaded {count_noun(loaded, 'AniList entry', 'AniList entries')} from cache")

    def save_cache(self, *, preview: bool) -> None:
        """Persist any newly seen AniList responses back to the on-disk cache.

        An entry that's already stored and still fresh keeps its original
        fetched_at (so the TTL actually expires it rather than resetting every
        run); a missing OR stale entry is (re)written with the current time, so
        an aged-out id is refreshed instead of being re-fetched on every run.

        Args:
            preview: When True, keep the warmed entries in memory but
                don't persist them (the gate lives in `CacheStore.save`).
        """

        now = datetime.now()
        now_str = now.strftime(UPDATED_AT_STR_FORMAT)
        cutoff = now - timedelta(days=ANILIST_CACHE_TTL_DAYS)

        written = 0
        for al_id, data in self.al_cache.items():
            if record_is_fresh(
                self._cache.get_anilist_meta(al_id),
                payload_key="data",
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
            self.logger.debug(f"Evicted {count_noun(evicted, 'stale AniList meta record')}")

    def prefetch(
        self,
        al_ids: Iterable[int],
        *,
        preview: bool,
        progress: ProgressSink | None = None,
    ) -> int:
        """Warm the AniList cache for a set of ids in batched requests.

        Fetches everything still missing from the cache in ANILIST_BATCH_SIZE-id
        "id_in" pages (one request per page) instead of one request per id on
        demand, then persists the results. This is what collapses a cold run's
        ~one-AniList-request-per-series into a handful, so the per-title loop
        rarely has to hit AniList one id at a time and trip its rate limit.

        Args:
            al_ids: Candidate AniList IDs for this run
            preview: Forwarded to the post-fetch save's preview gate.
            progress: Boot cockpit step fed per-batch
                fraction + "done/total" detail; None outside the cockpit.

        Returns:
            How many ids needed fetching (0 = fully cache-warm), for the
            caller's ledger detail.
        """

        missing = sorted(
            {i for i in al_ids if i not in self.al_cache},
        )
        total = len(missing)
        if not total:
            return 0

        done = 0
        for start in range(0, total, ANILIST_BATCH_SIZE):
            chunk = missing[start : start + ANILIST_BATCH_SIZE]
            # Ids unknown to AniList are simply absent from the result; the
            # per-id helpers will try once more on demand and degrade gracefully
            for al_id, data in self._client.query_batch(chunk).items():
                self.al_cache[al_id] = data
            done += len(chunk)
            if progress is not None:
                progress.progress(done / total, f"{done}/{total}")

        # Persist now (before the main loop) so the batch's work survives even an
        # early return - e.g., when max_torrents_to_add is hit mid-run
        self.save_cache(preview=preview)
        return total

    def _media(self, al_id: int) -> AniListMediaNode:
        """Resolve the typed Media node for an id: the run cache, then the wire.

        The get-or-fetch policy lives here, beside the cache it manages: a hit
        parses the stored body; a miss queries AniList and stores the raw body
        ONLY if it actually carried Media, so a transient failure (rate-limit,
        network) isn't cached as a permanent miss - the next call gets a fresh
        chance.

        Returns:
            The parsed `AniListMediaNode`; all-`None` on a miss.
        """

        # Cache hit: parse the stored body's Media node.
        body = self.al_cache.get(al_id)
        if body is not None:
            return media_from(body)

        # Miss: query AniList. Extract the raw Media dict once to gate the cache
        # store, then parse it into the typed node for the return. The cached
        # body is only ever read, so store it directly rather than copying.
        fetched = self._client.query(al_id)
        raw_media = extract_path(fetched, "data", "Media")
        if raw_media:
            self.al_cache[al_id] = fetched

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
        """Resolve the AniList episode count for an id, or None."""

        return self._media(al_id).episodes
