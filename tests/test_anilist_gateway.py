# pyright: strict
"""Contract tests for `AniListGateway`: the refresh age, stale serving, eviction, prefetch, and the per-id resolvers.

The gateway is driven over the in-memory `FakeCacheStore`. The HTTP/retry
layer is already pinned in `test_anilist_client`, so the wire is the shared
scripted `AniListClient` injected at construction.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from pearlarr.anilist_client import ANILIST_BATCH_SIZE
from pearlarr.anilist_gateway import ANILIST_REFRESH_AGE_DAYS, AniListGateway
from pearlarr.cache import UPDATED_AT_STR_FORMAT

from .builders import FakeCacheStore, ScriptedAniListClient, anilist_body, make_anilist_gateway
from .fakes import CaptureHandler


def _stamp(*, days_ago: int) -> str:
    """A `fetched_at` stamp `days_ago` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _now_str() -> str:
    """The current time in the store's stamp format, for "written now" assertions."""

    return datetime.now().strftime(UPDATED_AT_STR_FORMAT)


def _fresh_record(al_id: int) -> dict[str, Any]:
    """A populated store record stamped inside the refresh age."""

    return {"fetched_at": _stamp(days_ago=1), "data": anilist_body(al_id)}


def _stale_record(al_id: int) -> dict[str, Any]:
    """A populated store record stamped past the refresh age."""

    return {"fetched_at": _stamp(days_ago=ANILIST_REFRESH_AGE_DAYS + 1), "data": anilist_body(al_id)}


class _RecordingSink:
    """Typed recording ProgressSink: captures each (fraction, detail) update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.updates.append((fraction, detail))


class TestLoadCache:
    """load_cache seeds every stored record, including the ones past the refresh age."""

    def test_every_payload_loads_whatever_its_stamp(self) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _fresh_record(1))
        store.put_anilist_meta(2, _stale_record(2))
        store.put_anilist_meta(3, {"data": anilist_body(3)})  # stamp-less -> unreadable age
        store.put_anilist_meta(4, {"fetched_at": _stamp(days_ago=1)})  # no payload -> nothing to serve

        gateway.load_cache()

        assert gateway.al_cache == {1: anilist_body(1), 2: anilist_body(2), 3: anilist_body(3)}
        # A week-old count beats withholding it: None downstream means "use every episode".
        assert gateway.n_eps(2) == 12
        assert client.query_calls == []


class TestPrefetch:
    """prefetch batches the missing and stale ids, persists exactly what it fetched, and stops on an outage."""

    def test_all_fresh_fetches_nothing(self) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, _fresh_record(al_id))
        gateway.load_cache()

        assert gateway.prefetch([1, 2], preview=True) == 0
        assert client.batch_calls == []

    def test_51_missing_chunks_merges_persists_and_reports(self) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]). id 7 is unknown to AniList
        # (absent from the batch result), which is just a miss, never a raise.
        client = ScriptedAniListClient(absent=frozenset({7}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=False, progress=sink)

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

    def test_writes_exactly_the_fetched_ids(self) -> None:
        client = ScriptedAniListClient(absent=frozenset({2}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        original = _stamp(days_ago=1)
        store.put_anilist_meta(1, {"fetched_at": original, "data": anilist_body(1)})
        gateway.load_cache()
        # A confirmed miss is remembered for the run, so the prefetch has nothing to ask about it.
        assert gateway.title(2) is None

        before = _now_str()
        gateway.prefetch([1, 2, 3], preview=False)

        assert client.batch_calls == [[3]]
        # A loaded fresh row keeps its stamp, so the refresh age can actually reach it.
        fresh = store.get_anilist_meta(1)
        assert fresh is not None
        assert fresh["fetched_at"] == original
        # The remembered miss never reaches disk.
        assert store.get_anilist_meta(2) is None
        written = store.get_anilist_meta(3)
        assert written is not None
        assert written["fetched_at"] >= before
        assert written["data"] == anilist_body(3)

    def test_a_preview_save_writes_nothing(self) -> None:
        # A preview never commits, so the fetch stays in memory and no aged row is evicted.
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _stale_record(1))

        gateway.prefetch([3], preview=True)

        assert client.batch_calls == [[3]]
        assert gateway.al_cache[3] == anilist_body(3)
        assert store.get_anilist_meta(3) is None
        assert store.get_anilist_meta(1) is not None

    def test_stops_once_the_breaker_trips(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=False, progress=sink)

        assert fetched == len(ids)
        assert gateway.outage is True
        # The first chunk tripped the breaker, so the second was never requested.
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE]]
        # The tripped chunk fetched nothing, so the cockpit never claimed it as progress.
        assert sink.updates == []
        assert gateway.al_cache == {}
        assert store.get_anilist_meta(1) is None

    def test_outage_keeps_stale_rows_and_serves_them(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _stale_record(1))
        store.put_anilist_meta(9, _stale_record(9))
        gateway.load_cache()

        gateway.prefetch([1], preview=False)

        assert gateway.outage is True
        # Nothing evicted: the aged rows are what is serving this run.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(9) is not None
        assert gateway.title(1) == "Resolved"
        # The aged count still answers: the alternative is grabbing every episode.
        assert gateway.n_eps(1) == 12
        assert client.query_calls == []

    def test_healthy_refresh_writes_and_evicts(self) -> None:
        client = ScriptedAniListClient(absent=frozenset({2}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
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
        # Not returned: still served from memory this run, but evicted from disk.
        assert store.get_anilist_meta(2) is None
        assert gateway.title(2) == "Resolved"
        assert gateway.n_eps(2) == 12
        assert client.query_calls == []


class TestMediaResolution:
    """The per-id resolvers: get-or-fetch against the run cache, typed reads, misses remembered."""

    def test_resolvers_read_typed_fields_and_cache_the_fetch(self) -> None:
        client = ScriptedAniListClient()
        gateway = make_anilist_gateway(client)

        assert gateway.title(42) == "Resolved"
        assert gateway.thumb(42) == "https://img/large"
        assert gateway.banner(42) == "https://img/banner"
        assert gateway.media_format(42) == "TV"
        assert gateway.n_eps(42) == 12
        # The first resolver fetched and stored the raw body. The other four
        # were cache hits, so the wire saw exactly one query.
        assert client.query_calls == [42]
        assert 42 in gateway.al_cache

    def test_title_prefers_english_then_romaji(self) -> None:
        gateway = make_anilist_gateway()
        gateway.al_cache = {
            1: {"data": {"Media": {"id": 1, "title": {"english": "E", "romaji": "R"}}}},
            2: {"data": {"Media": {"id": 2, "title": {"romaji": "R"}}}},
            3: {"data": {"Media": {"id": 3}}},
        }

        assert gateway.title(1) == "E"
        assert gateway.title(2) == "R"
        assert gateway.title(3) is None

    def test_definite_miss_is_remembered_for_the_run(self) -> None:
        # AniList answered with no Media: the id is unknown, so no later accessor asks again.
        client = ScriptedAniListClient(absent=frozenset({7}))
        gateway = make_anilist_gateway(client)

        assert gateway.title(7) is None
        assert gateway.n_eps(7) is None

        assert gateway.al_cache[7] == {}
        assert client.query_calls == [7]

    def test_failure_trips_the_breaker_without_remembering_a_miss(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({7}))
        gateway = make_anilist_gateway(client)

        assert gateway.title(7) is None
        assert gateway.outage is True
        # No answer is not a miss: the id stays unknown instead of remembered as absent.
        assert gateway.al_cache == {}


class TestLogPluralization:
    """The debug ledger lines pluralize by count (no "1 entries" / manual "(s)")."""

    def _captured(self, gateway: AniListGateway) -> CaptureHandler:
        handler = CaptureHandler()
        gateway.logger.handlers = [handler]
        gateway.logger.setLevel(logging.DEBUG)
        return handler

    def test_load_cache_singular(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        store.put_anilist_meta(1, _fresh_record(1))
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 1 AniList entry from cache"]

    def test_load_cache_plural(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, _fresh_record(al_id))
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 2 AniList entries from cache"]

    def test_evict_singular(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        store.put_anilist_meta(1, _stale_record(1))
        handler = self._captured(gateway)

        gateway.save_cache(preview=False)

        assert [r.getMessage().strip() for r in handler.records] == ["Evicted 1 stale AniList meta record"]
