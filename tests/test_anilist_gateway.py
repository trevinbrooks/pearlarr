# pyright: strict
"""Contract tests for `AniListGateway`: the refresh age, stale serving, eviction, prefetch, and the per-id resolvers.

The gateway is driven over the in-memory `FakeCacheStore`. The HTTP/retry
layer is already pinned in `test_anilist_client`, so the wire is faked with a
checked scripted `AniListClient` subclass injected at construction.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, override

import httpx

from pearlarr.anilist_client import ANILIST_BATCH_SIZE, MAX_RETRIES, AniListCache, AniListClient
from pearlarr.anilist_gateway import ANILIST_CACHE_TTL_DAYS, AniListGateway
from pearlarr.cache import UPDATED_AT_STR_FORMAT

from .builders import FakeCacheStore, make_logger
from .fakes import CaptureHandler


def _stamp(*, days_ago: int) -> str:
    """A `fetched_at` stamp `days_ago` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _now_str() -> str:
    """The current time in the store's stamp format, for "written now" assertions."""

    return datetime.now().strftime(UPDATED_AT_STR_FORMAT)


def _media(al_id: int) -> dict[str, dict[str, Any]]:
    """A cached AniList body for `al_id`: the run-cache value, and the stored record's `data`."""

    return {"data": {"Media": {"id": al_id}}}


def _full_media(al_id: int) -> dict[str, dict[str, Any]]:
    """A fully-populated body, for the per-id field-resolution tests."""

    return {
        "data": {
            "Media": {
                "id": al_id,
                "title": {"english": "English Title", "romaji": "Romaji Title"},
                "coverImage": {"large": "https://img/large"},
                "bannerImage": "https://img/banner",
                "episodes": 12,
                "format": "TV",
            },
        },
    }


def _stale_record(al_id: int) -> dict[str, Any]:
    """A fully-populated store record stamped past the refresh age."""

    return {"fetched_at": _stamp(days_ago=ANILIST_CACHE_TTL_DAYS + 1), "data": _full_media(al_id)}


class _ScriptedClient(AniListClient):
    """Checked scripted `AniListClient`: canned bodies, calls recorded, no HTTP.

    `absent` ids are unknown to AniList (a Media-less answer). A request touching a
    `failing` id trips the breaker and answers empty. `full` scripts populated bodies.
    """

    def __init__(
        self,
        absent: frozenset[int] = frozenset(),
        *,
        failing: frozenset[int] = frozenset(),
        full: bool = False,
    ) -> None:
        # The real ctor is network-free: it just binds the client.
        super().__init__(client=httpx.Client())
        self._absent = absent
        self._failing = failing
        self._full = full
        self.query_calls: list[int] = []
        self.batch_calls: list[list[int]] = []

    def _body(self, al_id: int) -> dict[str, dict[str, Any]]:
        return _full_media(al_id) if self._full else _media(al_id)

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        self.query_calls.append(al_id)
        if self.outage:
            return {}
        if al_id in self._failing:
            self._note_outage(f"request failed after {MAX_RETRIES} retries")
            return {}
        if al_id in self._absent:
            return {"data": {"Media": None}}
        return self._body(al_id)

    @override
    def query_batch(self, al_ids: list[int]) -> AniListCache:
        self.batch_calls.append(list(al_ids))
        if self.outage:
            return {}
        if self._failing & set(al_ids):
            self._note_outage(f"request failed after {MAX_RETRIES} retries")
            return {}
        return {i: self._body(i) for i in al_ids if i not in self._absent}


def _make_gateway(client: AniListClient | None = None) -> tuple[AniListGateway, FakeCacheStore]:
    store = FakeCacheStore()
    return AniListGateway(cache_store=store, logger=make_logger(), client=client or _ScriptedClient()), store


class _RecordingSink:
    """Typed recording ProgressSink: captures each (fraction, detail) update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.updates.append((fraction, detail))


class TestLoadCache:
    """load_cache seeds every stored record. Those past the refresh age serve, minus the episode count."""

    def test_every_payload_loads_and_aged_stamps_hold_back_the_count(self) -> None:
        client = _ScriptedClient()
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(1, {"fetched_at": _stamp(days_ago=1), "data": _full_media(1)})
        store.put_anilist_meta(2, _stale_record(2))
        store.put_anilist_meta(3, {"data": _full_media(3)})  # stamp-less -> unreadable age
        store.put_anilist_meta(4, {"fetched_at": _stamp(days_ago=1)})  # no payload -> nothing to serve

        gateway.load_cache()

        assert gateway.al_cache == {1: _full_media(1), 2: _full_media(2), 3: _full_media(3)}
        assert gateway.n_eps(1) == 12
        assert gateway.n_eps(2) is None
        assert gateway.n_eps(3) is None
        assert client.query_calls == []

    def test_stale_serves_every_field_but_the_count(self) -> None:
        client = _ScriptedClient()
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(2, _stale_record(2))

        gateway.load_cache()

        assert gateway.title(2) == "English Title"
        assert gateway.thumb(2) == "https://img/large"
        assert gateway.banner(2) == "https://img/banner"
        assert gateway.media_format(2) == "TV"
        # The count is the one field the refresh age exists for: None means "use every episode".
        assert gateway.n_eps(2) is None
        assert client.query_calls == []


class TestPrefetch:
    """prefetch batches the missing and stale ids, merges, persists, and stops on an outage."""

    def test_all_fresh_fetches_nothing(self) -> None:
        client = _ScriptedClient()
        gateway, store = _make_gateway(client)
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, {"fetched_at": _stamp(days_ago=1), "data": _media(al_id)})
        gateway.load_cache()

        assert gateway.prefetch([1, 2], preview=True) == 0
        assert client.batch_calls == []

    def test_51_missing_chunks_merges_persists_and_reports(self) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]). id 7 is unknown to AniList
        # (absent from the batch result), which is just a miss, never a raise.
        client = _ScriptedClient(absent=frozenset({7}))
        gateway, store = _make_gateway(client)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=True, progress=sink)

        # Returns how many NEEDED fetching (the absent id still counted as work).
        assert fetched == len(ids)
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE], ids[ANILIST_BATCH_SIZE:]]
        # Both pages merged into the run cache. The absent id is simply absent.
        assert set(gateway.al_cache) == set(ids) - {7}
        # Persisted (via put_anilist_meta) before returning, so the batch's work
        # survives an early run exit.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(ids[-1]) is not None
        assert store.get_anilist_meta(7) is None
        # The cockpit sink saw one (fraction, "done/total") update per batch.
        assert sink.updates == [(50 / 51, "50/51"), (1.0, "51/51")]

    def test_stale_ids_are_refetched(self) -> None:
        client = _ScriptedClient(full=True)
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(1, {"fetched_at": _stamp(days_ago=1), "data": _full_media(1)})
        store.put_anilist_meta(2, _stale_record(2))
        gateway.load_cache()
        assert gateway.n_eps(2) is None

        before = _now_str()
        assert gateway.prefetch([1, 2], preview=True) == 1

        assert client.batch_calls == [[2]]
        # Refreshed: the count serves again and the row carries a new stamp.
        assert gateway.n_eps(2) == 12
        record = store.get_anilist_meta(2)
        assert record is not None
        assert record["fetched_at"] >= before

    def test_stops_once_the_breaker_trips(self) -> None:
        client = _ScriptedClient(failing=frozenset({1}))
        gateway, store = _make_gateway(client)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=False, progress=sink)

        assert fetched == len(ids)
        assert gateway.outage is True
        # The first chunk tripped the breaker, so the second was never requested.
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE]]
        assert sink.updates == [(50 / 51, "50/51")]
        assert gateway.al_cache == {}
        assert store.get_anilist_meta(1) is None
        # Every later lookup answers without a request.
        assert gateway.title(ids[-1]) is None
        assert client.query_calls == []

    def test_outage_keeps_stale_rows_and_serves_them(self) -> None:
        client = _ScriptedClient(failing=frozenset({1}))
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(1, _stale_record(1))
        store.put_anilist_meta(9, _stale_record(9))
        gateway.load_cache()

        gateway.prefetch([1], preview=False)

        assert gateway.outage is True
        # Nothing evicted: the aged rows are what is serving this run.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(9) is not None
        assert gateway.title(1) == "English Title"
        assert gateway.n_eps(1) is None
        assert client.query_calls == []

    def test_healthy_refresh_writes_and_evicts(self) -> None:
        client = _ScriptedClient(absent=frozenset({2}), full=True)
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(1, _stale_record(1))
        store.put_anilist_meta(2, _stale_record(2))
        gateway.load_cache()

        before = _now_str()
        assert gateway.prefetch([1, 2], preview=False) == 2

        assert client.batch_calls == [[1, 2]]
        # Returned: rewritten with a fresh stamp, the count serves again.
        record = store.get_anilist_meta(1)
        assert record is not None
        assert record["fetched_at"] >= before
        assert gateway.n_eps(1) == 12
        # Not returned: still served stale this run, evicted from disk, never re-queried.
        assert store.get_anilist_meta(2) is None
        assert gateway.title(2) == "English Title"
        assert gateway.n_eps(2) is None
        assert client.query_calls == []


class TestSaveCache:
    """save_cache writes exactly what this run fetched, and evicts only on a healthy real save."""

    def test_writes_exactly_the_fetched_ids(self) -> None:
        client = _ScriptedClient()
        gateway, store = _make_gateway(client)
        original = _stamp(days_ago=1)
        store.put_anilist_meta(1, {"fetched_at": original, "data": _media(1)})
        gateway.load_cache()
        gateway.al_cache[2] = _media(2)  # in memory but never fetched, so never written

        before = _now_str()
        gateway.prefetch([1, 2, 3], preview=True)

        assert client.batch_calls == [[3]]
        # A loaded fresh row keeps its stamp, so the refresh age can actually reach it.
        fresh = store.get_anilist_meta(1)
        assert fresh is not None
        assert fresh["fetched_at"] == original
        assert store.get_anilist_meta(2) is None
        written = store.get_anilist_meta(3)
        assert written is not None
        assert written["fetched_at"] >= before
        assert written["data"] == _media(3)
        # The fetched set drained: a later save rewrites nothing.
        store.put_anilist_meta(3, {"fetched_at": original, "data": _media(3)})
        gateway.save_cache(preview=True)
        rewritten = store.get_anilist_meta(3)
        assert rewritten is not None
        assert rewritten["fetched_at"] == original

    def test_single_id_fetch_is_written(self) -> None:
        client = _ScriptedClient(full=True)
        gateway, store = _make_gateway(client)

        assert gateway.title(42) == "English Title"
        gateway.save_cache(preview=True)

        record = store.get_anilist_meta(42)
        assert record is not None
        assert record["data"] == _full_media(42)

    def test_real_save_evicts_past_cutoff(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(1, _stale_record(1))

        gateway.save_cache(preview=False)

        assert store.get_anilist_meta(1) is None

    def test_preview_leaves_stale_records(self) -> None:
        gateway, store = _make_gateway()
        store.put_anilist_meta(1, _stale_record(1))

        gateway.save_cache(preview=True)

        assert store.get_anilist_meta(1) is not None

    def test_outage_leaves_stale_records(self) -> None:
        client = _ScriptedClient(failing=frozenset({5}))
        gateway, store = _make_gateway(client)
        store.put_anilist_meta(1, _stale_record(1))
        assert gateway.title(5) is None
        assert gateway.outage is True

        gateway.save_cache(preview=False)

        assert store.get_anilist_meta(1) is not None


class TestMediaResolution:
    """The per-id resolvers: get-or-fetch against the run cache, typed reads, misses remembered."""

    def test_resolvers_read_typed_fields_and_cache_the_fetch(self) -> None:
        client = _ScriptedClient(full=True)
        gateway, _ = _make_gateway(client)

        assert gateway.title(42) == "English Title"
        assert gateway.thumb(42) == "https://img/large"
        assert gateway.banner(42) == "https://img/banner"
        assert gateway.media_format(42) == "TV"
        assert gateway.n_eps(42) == 12
        # The first resolver fetched and stored the raw body. The other four
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

    def test_definite_miss_is_remembered_for_the_run(self) -> None:
        # AniList answered with no Media: the id is unknown, so no accessor asks again.
        client = _ScriptedClient(absent=frozenset({7}))
        gateway, _ = _make_gateway(client)

        assert gateway.title(7) is None
        assert gateway.thumb(7) is None
        assert gateway.banner(7) is None
        assert gateway.media_format(7) is None
        assert gateway.n_eps(7) is None

        assert 7 not in gateway.al_cache
        assert client.query_calls == [7]

    def test_prefetched_absent_ids_are_never_queried(self) -> None:
        client = _ScriptedClient(absent=frozenset({7}))
        gateway, _ = _make_gateway(client)
        gateway.prefetch([7], preview=True)

        assert gateway.title(7) is None
        assert gateway.thumb(7) is None
        assert gateway.banner(7) is None
        assert gateway.media_format(7) is None
        assert gateway.n_eps(7) is None

        assert client.query_calls == []

    def test_failure_trips_the_breaker_and_mutes_later_lookups(self) -> None:
        client = _ScriptedClient(failing=frozenset({7}), full=True)
        gateway, _ = _make_gateway(client)

        assert gateway.title(7) is None
        assert gateway.outage is True
        # Even an id AniList knows answers without a request now.
        assert gateway.title(8) is None
        assert client.query_calls == [7]


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
        store.put_anilist_meta(1, _stale_record(1))
        handler = self._captured(gateway)

        gateway.save_cache(preview=False)

        assert [r.getMessage().strip() for r in handler.records] == ["Evicted 1 stale AniList meta record"]
