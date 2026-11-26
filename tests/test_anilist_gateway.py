# pyright: strict
"""The AniList gateway: the refresh age, stale serving, eviction, the prefetch, the readers, and unread air dates.

The gateway is driven over the in-memory `FakeCacheStore`. The HTTP/retry
layer is already pinned in `test_anilist_client`, so the wire is the shared
scripted `AniListClient` injected at construction.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from pearlarr.anilist_client import ANILIST_BATCH_SIZE, AniListAiring
from pearlarr.anilist_gateway import ANILIST_REFRESH_AGE_DAYS
from pearlarr.stamps import STAMP_FORMAT

from .builders import FakeCacheStore, ScriptedAniListClient, anilist_body, make_anilist_gateway
from .fakes import RecordingProgress


def _stamp(*, days_ago: int) -> str:
    """A `fetched_at` stamp `days_ago` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(STAMP_FORMAT)


def _now_str() -> str:
    """The current time in the store's stamp format, for "written now" assertions."""

    return datetime.now().strftime(STAMP_FORMAT)


def _fresh_answer(anilist_id: int) -> dict[str, Any]:
    """A populated cached answer stamped inside the refresh age."""

    return {"fetched_at": _stamp(days_ago=1), "data": anilist_body(anilist_id)}


def _stale_answer(anilist_id: int) -> dict[str, Any]:
    """A populated cached answer stamped past the refresh age."""

    return {"fetched_at": _stamp(days_ago=ANILIST_REFRESH_AGE_DAYS + 1), "data": anilist_body(anilist_id)}


def _dateless_answer(anilist_id: int) -> dict[str, Any]:
    """A cached answer stamped inside the refresh age, with a Media that has no `startDate` key."""

    body = anilist_body(anilist_id)
    del body["data"]["Media"]["startDate"]
    return {"fetched_at": _stamp(days_ago=1), "data": body}


class TestPrefetch:
    """prefetch batches the missing and stale ids and persists exactly what it fetched.

    While AniList is unreachable it counts no progress and evicts nothing.
    """

    def test_all_fresh_fetches_nothing(self) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        for anilist_id in (1, 2):
            store.put_anilist_meta(anilist_id, _fresh_answer(anilist_id))
        gateway.load_cache()

        assert gateway.prefetch([1, 2], dry_run=False) == 0
        assert client.batch_calls == []

    def test_51_missing_chunks_persists_and_reports(self) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]). id 7 is unknown to AniList
        # (absent from the batch result), which is just not found, never a raise.
        client = ScriptedAniListClient(absent=frozenset({7}))
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        progress = RecordingProgress()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, dry_run=False, progress=progress)

        # Returns how many NEEDED fetching (the absent id still counted as work).
        assert fetched == len(ids)
        assert client.batch_calls == [ids[:ANILIST_BATCH_SIZE], ids[ANILIST_BATCH_SIZE:]]
        # Both pages' answers are stored. The absent id is simply absent.
        assert {anilist_id for anilist_id, _ in store.iter_anilist_meta()} == set(ids) - {7}
        # The progress handle saw one (fraction, "done/total") update per batch.
        assert progress.updates == [(50 / 51, "50/51"), (1.0, "51/51")]

    def test_writes_exactly_the_fetched_ids(self) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(ScriptedAniListClient(), store)
        loaded = _fresh_answer(1)
        store.put_anilist_meta(1, loaded)
        gateway.load_cache()

        gateway.prefetch([1, 3], dry_run=False)

        # A loaded fresh row keeps its stamp, so the refresh age can actually reach it.
        assert store.get_anilist_meta(1) == loaded
        written = store.get_anilist_meta(3)
        assert written is not None
        assert written["data"] == anilist_body(3)

    def test_a_dry_run_save_writes_nothing(self) -> None:
        # A dry run never commits: what it fetched isn't written, and no aged row is evicted.
        store = FakeCacheStore()
        gateway = make_anilist_gateway(ScriptedAniListClient(), store)
        aged = _stale_answer(1)
        store.put_anilist_meta(1, aged)

        gateway.prefetch([3], dry_run=True)

        assert dict(store.iter_anilist_meta()) == {1: aged}

    def test_what_anilist_answered_before_it_went_down_is_kept_and_counted(self) -> None:
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))
        store = FakeCacheStore()
        # AniList answers the first batch and fails on the second, which holds the last id alone.
        gateway = make_anilist_gateway(ScriptedAniListClient(failing=frozenset({ids[-1]})), store)
        progress = RecordingProgress()

        gateway.prefetch(ids, dry_run=False, progress=progress)

        # The next run doesn't ask again for the batch that came back.
        assert {anilist_id for anilist_id, _ in store.iter_anilist_meta()} == set(ids[:-1])
        # The boot step's bar claims only the batch AniList answered.
        assert progress.updates == [(50 / 51, "50/51")]

    @pytest.mark.parametrize(
        "stored",
        [pytest.param(_stale_answer(1), id="aged"), pytest.param(_dateless_answer(1), id="without-dates")],
    )
    def test_unreachable_keeps_stale_rows_and_serves_them(self, stored: dict[str, Any]) -> None:
        store = FakeCacheStore()
        gateway = make_anilist_gateway(ScriptedAniListClient(failing=frozenset({1})), store)
        store.put_anilist_meta(1, stored)
        gateway.load_cache()

        gateway.prefetch([1], dry_run=False)

        # Nothing evicted: the stored row is what serves this run.
        assert dict(store.iter_anilist_meta()) == {1: stored}
        # The stored count still answers: the alternative is grabbing every episode.
        assert gateway.episode_count(1) == 12

    @pytest.mark.parametrize("asked", [pytest.param([1, 2], id="asked-for"), pytest.param([2], id="not-asked-for")])
    def test_a_fresh_answer_stored_without_dates_is_refetched(self, asked: list[int]) -> None:
        client = ScriptedAniListClient()
        store = FakeCacheStore()
        gateway = make_anilist_gateway(client, store)
        store.put_anilist_meta(1, _dateless_answer(1))
        store.put_anilist_meta(2, _fresh_answer(2))
        # The contrast: aged but with its dates, so it waits until the run asks for its id.
        store.put_anilist_meta(3, _stale_answer(3))
        gateway.load_cache()

        gateway.prefetch(asked, dry_run=False)

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
        gateway.prefetch([1, 2], dry_run=False)

        assert client.batch_calls == [[1, 2]]
        # Returned: rewritten with a fresh stamp.
        answer = store.get_anilist_meta(1)
        assert answer is not None
        assert answer["fetched_at"] >= before
        # Not returned: evicted from disk, but still served from memory this run.
        assert store.get_anilist_meta(2) is None
        assert gateway.episode_count(2) == 12


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
        assert gateway.airing(42) == AniListAiring(
            start=date(2020, 1, 4), end=date(2020, 3, 21), episodes=12, duration=24
        )
        # The first resolver fetched and stored the raw body. The other five
        # were cache hits, so the wire saw exactly one query.
        assert client.query_calls == [42]

    @pytest.mark.parametrize(
        ("english", "romaji", "titles"),
        [
            pytest.param("E", "R", ("E", "R"), id="two-names"),
            pytest.param("Same", "Same", ("Same",), id="one-name-twice"),
            pytest.param(None, "R", ("R",), id="no-english-name"),
        ],
    )
    def test_titles_list_english_then_romaji_once_each(
        self, english: str | None, romaji: str, titles: tuple[str, ...]
    ) -> None:
        gateway = make_anilist_gateway()
        gateway.answers[1] = anilist_body(1, english, romaji)

        assert gateway.titles(1) == titles

    def test_definite_not_found_is_remembered_for_the_run(self) -> None:
        # AniList answered with no Media: the id is unknown, so no later accessor asks again.
        client = ScriptedAniListClient(absent=frozenset({7}))
        gateway = make_anilist_gateway(client)

        assert gateway.titles(7) == ()
        assert gateway.episode_count(7) is None
        assert client.query_calls == [7]


class TestUnread:
    """An id is unread only when an unreachable AniList left its lookup without a cached answer with its air dates."""

    @pytest.mark.parametrize(
        ("anilist_id", "unread"),
        [
            pytest.param(2, True, id="no-record"),
            pytest.param(3, True, id="stored-without-dates"),
            pytest.param(4, False, id="stored-with-dates"),
            pytest.param(5, False, id="answered-unknown"),
        ],
    )
    def test_an_unreachable_source_leaves_unread_a_looked_up_id_with_no_dated_answer(
        self, anilist_id: int, unread: bool
    ) -> None:
        store = FakeCacheStore()
        store.put_anilist_meta(3, _dateless_answer(3))
        store.put_anilist_meta(4, _stale_answer(4))
        gateway = make_anilist_gateway(ScriptedAniListClient(absent=frozenset({5}), failing=frozenset({1})), store)
        gateway.load_cache()
        # AniList answers that it doesn't have id 5, then fails on id 1.
        gateway.episode_count(5)
        gateway.episode_count(1)
        gateway.episode_count(anilist_id)

        assert gateway.unreachable is True
        assert gateway.air_dates_unread(anilist_id) is unread

    def test_an_id_nothing_looked_up_is_not_unread(self) -> None:
        gateway = make_anilist_gateway(ScriptedAniListClient(failing=frozenset({1})))
        gateway.episode_count(1)

        assert gateway.air_dates_unread(1) is True
        assert gateway.air_dates_unread(2) is False

    def test_an_answer_stored_without_dates_is_not_unread_while_anilist_answers(self) -> None:
        # The cached answer without dates serves the lookup, and AniList was never unreachable.
        store = FakeCacheStore()
        store.put_anilist_meta(3, _dateless_answer(3))
        gateway = make_anilist_gateway(ScriptedAniListClient(), store)
        gateway.load_cache()
        gateway.episode_count(3)

        assert gateway.air_dates_unread(3) is False
