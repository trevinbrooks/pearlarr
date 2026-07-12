# pyright: strict
"""Contract tests for `AniListGateway`: the TTL gates, eviction, prefetch, and the per-id get-or-fetch resolvers.

The gateway is driven over the in-memory `FakeCacheStore`; the HTTP/retry
layer is already pinned in `test_anilist_client`, so the wire is faked with a
checked scripted `AniListClient` subclass injected at construction.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, override

import httpx

from pearlarr.anilist_client import ANILIST_BATCH_SIZE, AniListCache, AniListClient
from pearlarr.anilist_gateway import ANILIST_CACHE_TTL_DAYS, AniListGateway
from pearlarr.cache import UPDATED_AT_STR_FORMAT

from .builders import FakeCacheStore, make_logger
from .fakes import CaptureHandler


def _stamp(*, days_ago: int) -> str:
    """A `fetched_at` stamp `days_ago` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _media(al_id: int) -> dict[str, dict[str, object]]:
    """A cached AniList body payload (the record's `data` value) for `al_id`."""

    return {"Media": {"id": al_id}}


def _full_media(al_id: int) -> dict[str, dict[str, object]]:
    """A fully-populated Media payload, for the per-id field-resolution tests."""

    return {
        "Media": {
            "id": al_id,
            "title": {"english": "English Title", "romaji": "Romaji Title"},
            "coverImage": {"large": "https://img/large"},
            "bannerImage": "https://img/banner",
            "episodes": 12,
            "format": "TV",
        },
    }


class _ScriptedClient(AniListClient):
    """Checked scripted `AniListClient`: canned bodies, calls recorded, no HTTP.

    Ids listed in `absent` are unknown to AniList: omitted from batch results
    and answered with a Media-less single-id body (a miss, never an error).
    `full` scripts fully-populated Media bodies instead of id-only ones.
    """

    def __init__(self, absent: frozenset[int] = frozenset(), *, full: bool = False) -> None:
        # The real ctor is network-free: it just binds the client.
        super().__init__(client=httpx.Client())
        self._absent = absent
        self._full = full
        self.query_calls: list[int] = []
        self.batch_calls: list[list[int]] = []

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        self.query_calls.append(al_id)
        if al_id in self._absent:
            return {"data": {"Media": None}}
        return {"data": _full_media(al_id) if self._full else _media(al_id)}

    @override
    def query_batch(self, al_ids: list[int]) -> AniListCache:
        self.batch_calls.append(list(al_ids))
        return {i: {"data": _media(i)} for i in al_ids if i not in self._absent}


def _make_gateway(client: AniListClient | None = None) -> tuple[AniListGateway, FakeCacheStore]:
    store = FakeCacheStore()
    return AniListGateway(cache_store=store, logger=make_logger(), client=client or _ScriptedClient()), store


class _RecordingSink:
    """Typed recording ProgressSink: captures each (fraction, detail) update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.updates.append((fraction, detail))


class TestLoadCacheTtlGate:
    """load_cache seeds only records still within the 7-day TTL."""

    def test_fresh_loads_stale_and_stampless_skipped(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(1, {"fetched_at": _stamp(days_ago=1), "data": _media(1)})
        store.put_anilist_meta(
            2,
            {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _media(2)},
        )
        store.put_anilist_meta(3, {"data": _media(3)})  # stamp-less -> unreadable age

        gateway.load_cache()

        assert gateway.al_cache == {1: _media(1)}


class TestSaveCacheFreshSkip:
    """save_cache keeps a still-fresh record's ORIGINAL stamp; rewrites the rest."""

    def test_fresh_keeps_stamp_stale_and_missing_rewritten(self) -> None:
        gateway, store = _make_gateway()
        original = _stamp(days_ago=1)
        store.put_anilist_meta(1, {"fetched_at": original, "data": _media(1)})
        store.put_anilist_meta(
            2,
            {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _media(2)},
        )
        gateway.al_cache = {1: _media(1), 2: _media(2), 3: _media(3)}

        before = datetime.now().strftime(UPDATED_AT_STR_FORMAT)
        gateway.save_cache(preview=True)

        # Still-fresh: the original stamp survives, so the TTL can actually expire
        # it (a per-run rewrite would reset the clock every run, forever).
        fresh = store.get_anilist_meta(1)
        assert fresh is not None
        assert fresh["fetched_at"] == original
        # Stale (2) and missing (3): (re)written with the current time.
        for al_id in (2, 3):
            record = store.get_anilist_meta(al_id)
            assert record is not None
            stamp = record["fetched_at"]
            assert isinstance(stamp, str)
            assert stamp >= before
            assert record["data"] == _media(al_id)


class TestSaveCachePreviewGate:
    """Eviction of aged-out records runs only on a real (non-preview) save."""

    def test_real_save_evicts_past_cutoff(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(
            1,
            {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _media(1)},
        )

        gateway.save_cache(preview=False)

        assert store.get_anilist_meta(1) is None

    def test_preview_leaves_stale_records(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(
            1,
            {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _media(1)},
        )

        gateway.save_cache(preview=True)

        assert store.get_anilist_meta(1) is not None


class TestPrefetch:
    """prefetch batches only the missing ids, merges, persists, and reports."""

    def test_all_cached_fetches_nothing(self) -> None:
        client = _ScriptedClient()
        gateway, _ = _make_gateway(client)
        gateway.al_cache = {1: _media(1), 2: _media(2)}

        assert gateway.prefetch([1, 2], preview=True) == 0
        assert client.batch_calls == []

    def test_51_missing_chunks_merges_persists_and_reports(self) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]); id 7 is unknown to AniList
        # (absent from the batch result), which is just a miss, never a raise.
        client = _ScriptedClient(absent=frozenset({7}))
        gateway, store = _make_gateway(client)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=True, progress=sink)

        # Returns how many NEEDED fetching (the absent id still counted as work).
        assert fetched == len(ids)
        # Both pages rode the injected wire client (which binds the transport and
        # the retry narration itself - no per-call threading left to mis-wire).
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE], ids[ANILIST_BATCH_SIZE:]]
        # Both pages merged into the run cache; the absent id is simply absent.
        assert set(gateway.al_cache) == set(ids) - {7}
        # Persisted (via put_anilist_meta) before returning, so the batch's work
        # survives an early run exit.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(ids[-1]) is not None
        assert store.get_anilist_meta(7) is None
        # The cockpit sink saw one (fraction, "done/total") update per batch.
        assert sink.updates == [(50 / 51, "50/51"), (1.0, "51/51")]


class TestMediaResolution:
    """The per-id resolvers: get-or-fetch against the run cache, typed reads.

    This is the policy that moved up from the old free-function layer to live
    beside the cache: check the run cache first, store a fetched body only when
    it actually carried Media, and read each resolver's field off the parsed
    node.
    """

    def test_resolvers_read_typed_fields_and_cache_the_fetch(self) -> None:
        client = _ScriptedClient(full=True)
        gateway, _ = _make_gateway(client)

        assert gateway.title(42) == "English Title"
        assert gateway.thumb(42) == "https://img/large"
        assert gateway.banner(42) == "https://img/banner"
        assert gateway.media_format(42) == "TV"
        assert gateway.n_eps(42) == 12
        # The first resolver fetched and stored the raw body; the other four
        # were cache hits, so the wire saw exactly one query.
        assert client.query_calls == [42]
        assert 42 in gateway.al_cache

    def test_title_prefers_english_then_romaji(self) -> None:
        gateway, _ = _make_gateway()
        gateway.al_cache = {
            1: {"data": {"Media": {"id": 1, "title": {"english": "E", "romaji": "R"}}}},
            2: {"data": {"Media": {"id": 2, "title": {"romaji": "R"}}}},
            3: {"data": {"Media": {"id": 3}}},
        }

        assert gateway.title(1) == "E"
        assert gateway.title(2) == "R"
        assert gateway.title(3) is None

    def test_transient_miss_not_cached_and_retried_next_call(self) -> None:
        # A Media-less body (unknown id, or a rate-limit that exhausted its
        # retries) must not be remembered as a permanent miss for the run.
        client = _ScriptedClient(absent=frozenset({7}))
        gateway, _ = _make_gateway(client)

        assert gateway.title(7) is None
        assert 7 not in gateway.al_cache
        assert gateway.n_eps(7) is None
        # The second resolver queried again - the miss got a fresh chance.
        assert client.query_calls == [7, 7]


class TestLogPluralization:
    """The debug ledger lines pluralize by count (no "1 entries" / manual "(s)")."""

    def _captured(self, gateway: AniListGateway) -> CaptureHandler:
        handler = CaptureHandler()
        gateway.logger.handlers = [handler]
        gateway.logger.setLevel(logging.DEBUG)
        return handler

    def test_load_cache_singular(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(1, {"fetched_at": _stamp(days_ago=1), "data": _media(1)})
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 1 AniList entry from cache"]

    def test_load_cache_plural(self) -> None:
        gateway, store = _make_gateway()
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, {"fetched_at": _stamp(days_ago=1), "data": _media(al_id)})
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 2 AniList entries from cache"]

    def test_evict_singular(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(
            1,
            {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _media(1)},
        )
        handler = self._captured(gateway)

        gateway.save_cache(preview=False)

        assert [r.getMessage().strip() for r in handler.records] == ["Evicted 1 stale AniList meta record"]
