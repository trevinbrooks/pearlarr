# pyright: strict
"""Contract tests for `AniListGateway`: the refresh age, stale serving, eviction, prefetch, and the per-id resolvers.

The gateway is driven over the in-memory `FakeCacheStore`. The HTTP/retry
layer is already pinned in `test_anilist_client`, so the wire is the shared
scripted `AniListClient` injected at construction.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pytest

from pearlarr.anilist_client import ANILIST_BATCH_SIZE, AniListAiring
from pearlarr.anilist_gateway import ANILIST_REFRESH_AGE_DAYS, AniListGateway
from pearlarr.stamps import STAMP_FORMAT

from .builders import FakeCacheStore, ScriptedAniListClient, anilist_body, make_anilist_gateway
from .fakes import CaptureHandler


def _stamp(*, days_ago: int) -> str:
    """A `fetched_at` stamp `days_ago` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(STAMP_FORMAT)


def _now_str() -> str:
    """The current time in the store's stamp format, for "written now" assertions."""

    return datetime.now().strftime(STAMP_FORMAT)


def _fresh_answer(al_id: int) -> dict[str, Any]:
    """A populated cached answer stamped inside the refresh age."""

    return {"fetched_at": _stamp(days_ago=1), "data": anilist_body(al_id)}


def _stale_answer(al_id: int) -> dict[str, Any]:
    """A populated cached answer stamped past the refresh age."""

    return {"fetched_at": _stamp(days_ago=ANILIST_REFRESH_AGE_DAYS + 1), "data": anilist_body(al_id)}


def _dateless_answer(al_id: int) -> dict[str, Any]:
    """A cached answer stamped inside the refresh age, with a Media that has no `startDate` key."""

    body = anilist_body(al_id)
    del body["data"]["Media"]["startDate"]
    return {"fetched_at": _stamp(days_ago=1), "data": body}


class _RecordingProgress:
    """Typed recording ProgressSink: captures each (fraction, detail) update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.updates.append((fraction, detail))


class TestLoadCache:
    """load_cache loads every cached answer, including the ones past the refresh age."""

    def test_every_payload_loads_whatever_its_stamp(self) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _fresh_answer(1))
        store.put_anilist_meta(2, _stale_answer(2))
        store.put_anilist_meta(3, {"data": anilist_body(3)})  # stamp-less -> unreadable age
        store.put_anilist_meta(4, {"fetched_at": _stamp(days_ago=1)})  # no payload -> nothing to serve

        gateway.load_cache()

        assert gateway.al_cache == {1: anilist_body(1), 2: anilist_body(2), 3: anilist_body(3)}
        # A week-old count beats withholding it: None downstream means "use every episode".
        assert gateway.episode_count(2) == 12
        assert client.query_calls == []


class TestPrefetch:
    """prefetch batches the missing and stale ids and persists exactly what it fetched.

    It stops when AniList is unreachable.
    """

    def test_all_fresh_fetches_nothing(self) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, _fresh_answer(al_id))
        gateway.load_cache()

        assert gateway.prefetch([1, 2], dry_run=True) == 0
        assert client.batch_calls == []

    def test_51_missing_chunks_merges_persists_and_reports(self) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]). id 7 is unknown to AniList
        # (absent from the batch result), which is just not found, never a raise.
        client = ScriptedAniListClient(absent=frozenset({7}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        progress = _RecordingProgress()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, dry_run=False, progress=progress)

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
        # The progress handle saw one (fraction, "done/total") update per batch.
        assert progress.updates == [(50 / 51, "50/51"), (1.0, "51/51")]

    def test_writes_exactly_the_fetched_ids(self) -> None:
        client = ScriptedAniListClient(absent=frozenset({2}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        original = _stamp(days_ago=1)
        store.put_anilist_meta(1, {"fetched_at": original, "data": anilist_body(1)})
        gateway.load_cache()
        # A confirmed not-found answer is kept for the run, so the prefetch has nothing to ask about it.
        assert gateway.titles(2) == ()

        before = _now_str()
        gateway.prefetch([1, 2, 3], dry_run=False)

        assert client.batch_calls == [[3]]
        # A loaded fresh row keeps its stamp, so the refresh age can actually reach it.
        fresh = store.get_anilist_meta(1)
        assert fresh is not None
        assert fresh["fetched_at"] == original
        # The not-found answer kept for the run never reaches disk.
        assert store.get_anilist_meta(2) is None
        written = store.get_anilist_meta(3)
        assert written is not None
        assert written["fetched_at"] >= before
        assert written["data"] == anilist_body(3)

    def test_a_preview_save_writes_nothing(self) -> None:
        # A dry run never commits, so the fetch stays in memory and no aged row is evicted.
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _stale_answer(1))

        gateway.prefetch([3], dry_run=True)

        assert client.batch_calls == [[3]]
        assert gateway.al_cache[3] == anilist_body(3)
        assert store.get_anilist_meta(3) is None
        assert store.get_anilist_meta(1) is not None

    def test_stops_once_the_breaker_trips(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        progress = _RecordingProgress()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, dry_run=False, progress=progress)

        assert fetched == len(ids)
        assert gateway.unreachable is True
        # The first chunk marked AniList unreachable, so the second was never requested.
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE]]
        # The tripped chunk fetched nothing, so the boot step's bar never claimed it as progress.
        assert progress.updates == []
        assert gateway.al_cache == {}
        assert store.get_anilist_meta(1) is None

    def test_outage_keeps_stale_rows_and_serves_them(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _stale_answer(1))
        store.put_anilist_meta(9, _stale_answer(9))
        gateway.load_cache()

        gateway.prefetch([1], dry_run=False)

        assert gateway.unreachable is True
        # Nothing evicted: the aged rows are what is serving this run.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(9) is not None
        assert gateway.titles(1) == ("Resolved",)
        # The aged count still answers: the alternative is grabbing every episode.
        assert gateway.episode_count(1) == 12
        assert client.query_calls == []

    @pytest.mark.parametrize("asked", [pytest.param([1, 2], id="asked-for"), pytest.param([2], id="not-asked-for")])
    def test_a_fresh_record_stored_without_dates_is_refetched(self, asked: list[int]) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _dateless_answer(1))
        store.put_anilist_meta(2, _fresh_answer(2))
        gateway.load_cache()

        assert gateway.prefetch(asked, dry_run=False) == 1

        assert client.batch_calls == [[1]]
        written = store.get_anilist_meta(1)
        assert written is not None
        assert written["data"] == anilist_body(1)

    def test_healthy_refresh_writes_and_evicts(self) -> None:
        client = ScriptedAniListClient(absent=frozenset({2}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _stale_answer(1))
        store.put_anilist_meta(2, _stale_answer(2))
        gateway.load_cache()

        before = _now_str()
        assert gateway.prefetch([1, 2], dry_run=False) == 2

        assert client.batch_calls == [[1, 2]]
        # Returned: rewritten with a fresh stamp, the count serves again.
        answer = store.get_anilist_meta(1)
        assert answer is not None
        assert answer["fetched_at"] >= before
        assert gateway.episode_count(1) == 12
        # Not returned: still served from memory this run, but evicted from disk.
        assert store.get_anilist_meta(2) is None
        assert gateway.titles(2) == ("Resolved",)
        assert gateway.episode_count(2) == 12
        assert client.query_calls == []


class TestMediaResolution:
    """The per-id resolvers: get-or-fetch against the run cache, typed reads, not-found answers kept for the run."""

    def test_resolvers_read_typed_fields_and_cache_the_fetch(self) -> None:
        client = ScriptedAniListClient()
        gateway = make_anilist_gateway(client)

        assert gateway.titles(42) == ("Resolved",)
        assert gateway.thumb(42) == "https://img/large"
        assert gateway.banner(42) == "https://img/banner"
        assert gateway.media_format(42) == "TV"
        assert gateway.episode_count(42) == 12
        # The first resolver fetched and stored the raw body. The other four
        # were cache hits, so the wire saw exactly one query.
        assert client.query_calls == [42]
        assert 42 in gateway.al_cache

    def test_airing_reads_the_dates_count_and_episode_length(self) -> None:
        gateway = make_anilist_gateway()
        gateway.al_cache = {
            1: anilist_body(1),
            2: {"data": {"Media": {"id": 2, "episodes": 12}}},
        }

        assert gateway.airing(1) == AniListAiring(
            start=date(2020, 1, 4), end=date(2020, 3, 21), episodes=12, duration=24
        )
        assert gateway.airing(2) == AniListAiring(start=None, end=None, episodes=12, duration=None)

    def test_titles_list_english_then_romaji_once_each(self) -> None:
        gateway = make_anilist_gateway()
        gateway.al_cache = {
            1: {"data": {"Media": {"id": 1, "title": {"english": "E", "romaji": "R"}}}},
            2: {"data": {"Media": {"id": 2, "title": {"english": "Same", "romaji": "Same"}}}},
            3: {"data": {"Media": {"id": 3, "title": {"romaji": "R"}}}},
            4: {"data": {"Media": {"id": 4}}},
        }

        assert gateway.titles(1) == ("E", "R")
        assert gateway.titles(2) == ("Same",)
        assert gateway.titles(3) == ("R",)
        assert gateway.titles(4) == ()

    def test_definite_miss_is_remembered_for_the_run(self) -> None:
        # AniList answered with no Media: the id is unknown, so no later accessor asks again.
        client = ScriptedAniListClient(absent=frozenset({7}))
        gateway = make_anilist_gateway(client)

        assert gateway.titles(7) == ()
        assert gateway.episode_count(7) is None

        assert gateway.al_cache[7] == {}
        assert client.query_calls == [7]

    def test_failure_trips_the_breaker_without_remembering_a_miss(self) -> None:
        client = ScriptedAniListClient(failing=frozenset({7}))
        gateway = make_anilist_gateway(client)

        assert gateway.titles(7) == ()
        assert gateway.unreachable is True
        # No answer is not a not-found answer: the id stays unknown instead of kept as absent.
        assert gateway.al_cache == {}


class TestUnread:
    """An id is unread only when an unreachable AniList left its lookup without a cached answer with its air dates."""

    @pytest.mark.parametrize(
        ("al_id", "unread"),
        [
            pytest.param(2, True, id="no-record"),
            pytest.param(3, True, id="stored-without-dates"),
            pytest.param(4, False, id="stored-with-dates"),
            pytest.param(5, False, id="answered-unknown"),
        ],
    )
    def test_an_outage_leaves_unread_a_looked_up_id_with_no_dated_record(self, al_id: int, unread: bool) -> None:
        store = FakeCacheStore()
        store.put_anilist_meta(3, _dateless_answer(3))
        store.put_anilist_meta(4, _stale_answer(4))
        gateway = make_anilist_gateway(ScriptedAniListClient(absent=frozenset({5}), failing=frozenset({1})), store)
        gateway.load_cache()
        # AniList answers that it doesn't have id 5, then fails on id 1.
        gateway.episode_count(5)
        gateway.episode_count(1)
        gateway.episode_count(al_id)

        assert gateway.unreachable is True
        assert gateway.air_dates_unread(al_id) is unread

    def test_an_id_nothing_looked_up_is_not_unread(self) -> None:
        gateway = make_anilist_gateway(ScriptedAniListClient(failing=frozenset({1})))
        gateway.episode_count(1)

        assert gateway.air_dates_unread(1) is True
        assert gateway.air_dates_unread(2) is False

    def test_a_record_stored_without_dates_is_not_unread_while_anilist_answers(self) -> None:
        # The cached answer without dates serves the lookup, and AniList was never unreachable.
        store = FakeCacheStore()
        store.put_anilist_meta(3, _dateless_answer(3))
        gateway = make_anilist_gateway(ScriptedAniListClient(), store)
        gateway.load_cache()
        gateway.episode_count(3)

        assert gateway.air_dates_unread(3) is False


class TestLogPluralization:
    """The debug lines pluralize by count (no "1 entries" / manual "(s)")."""

    def _captured(self, gateway: AniListGateway) -> CaptureHandler:
        handler = CaptureHandler()
        gateway.logger.handlers = [handler]
        gateway.logger.setLevel(logging.DEBUG)
        return handler

    def test_load_cache_singular(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        store.put_anilist_meta(1, _fresh_answer(1))
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 1 AniList entry from cache"]

    def test_load_cache_plural(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        for al_id in (1, 2):
            store.put_anilist_meta(al_id, _fresh_answer(al_id))
        handler = self._captured(gateway)

        gateway.load_cache()

        assert [r.getMessage().strip() for r in handler.records] == ["Loaded 2 AniList entries from cache"]

    def test_evict_singular(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(store=store)
        store.put_anilist_meta(1, _stale_answer(1))
        handler = self._captured(gateway)

        gateway.save_cache(dry_run=False)

        assert [r.getMessage().strip() for r in handler.records] == ["Evicted 1 stale AniList meta record"]
