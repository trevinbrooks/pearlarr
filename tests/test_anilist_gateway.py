# pyright: strict
"""Contract tests for ``AniListGateway``: the TTL gates, eviction, and prefetch.

The gateway is driven over the in-memory :class:`FakeCacheStore`; the HTTP/retry
layer (``get_query_batch``'s transport) is already pinned in ``test_anilist``,
so prefetch fakes it with a recording batch stub at the module attribute.
"""

import logging
from datetime import datetime, timedelta

import pytest

import seadexarr.modules.anilist_gateway as anilist_gateway
from seadexarr.modules.anilist import ANILIST_BATCH_SIZE, AniListCache, AniListRetryLog
from seadexarr.modules.anilist_gateway import ANILIST_CACHE_TTL_DAYS, AniListGateway
from seadexarr.modules.cache import UPDATED_AT_STR_FORMAT

from .builders import FakeCacheStore, make_logger
from .fakes import CaptureHandler


def _stamp(*, days_ago: int) -> str:
    """A ``fetched_at`` stamp ``days_ago`` days in the past (store string format)."""

    return (datetime.now() - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _media(al_id: int) -> dict[str, dict[str, object]]:
    """A cached AniList body payload (the record's ``data`` value) for ``al_id``."""

    return {"Media": {"id": al_id}}


def _make_gateway() -> tuple[AniListGateway, FakeCacheStore]:
    store = FakeCacheStore()
    return AniListGateway(cache_store=store, logger=make_logger()), store


class _BatchRecorder:
    """Recording stand-in for ``get_query_batch``: scripted per-id bodies, no HTTP.

    Ids listed in ``absent`` are omitted from the result, mirroring AniList's
    behaviour for ids it doesn't know (simply missing, never an error).
    """

    def __init__(self, absent: frozenset[int] = frozenset()) -> None:
        self._absent = absent
        self.calls: list[list[int]] = []
        self.retry_logs: list[AniListRetryLog | None] = []

    def __call__(self, al_ids: list[int], retry_log: AniListRetryLog | None = None) -> AniListCache:
        self.calls.append(list(al_ids))
        self.retry_logs.append(retry_log)
        return {i: {"data": _media(i)} for i in al_ids if i not in self._absent}


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

    def test_all_cached_fetches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway, _ = _make_gateway()
        recorder = _BatchRecorder()
        monkeypatch.setattr(anilist_gateway, "get_query_batch", recorder)
        gateway.al_cache = {1: _media(1), 2: _media(2)}

        assert gateway.prefetch([1, 2], preview=True) == 0
        assert recorder.calls == []

    def test_51_missing_chunks_merges_persists_and_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 51 missing ids -> two id_in pages ([50, 1]); id 7 is unknown to AniList
        # (absent from the batch result), which is just a miss, never a raise.
        gateway, store = _make_gateway()
        recorder = _BatchRecorder(absent=frozenset({7}))
        monkeypatch.setattr(anilist_gateway, "get_query_batch", recorder)
        sink = _RecordingSink()
        ids = list(range(1, ANILIST_BATCH_SIZE + 2))

        fetched = gateway.prefetch(ids, preview=True, progress=sink)

        # Returns how many NEEDED fetching (the absent id still counted as work).
        assert fetched == len(ids)
        assert recorder.calls == [ids[:ANILIST_BATCH_SIZE], ids[ANILIST_BATCH_SIZE:]]
        # Both pages merged into the run cache; the absent id is simply absent.
        assert set(gateway.al_cache) == set(ids) - {7}
        # Persisted (via put_anilist_meta) before returning, so the batch's work
        # survives an early run exit.
        assert store.get_anilist_meta(1) is not None
        assert store.get_anilist_meta(ids[-1]) is not None
        assert store.get_anilist_meta(7) is None
        # The cockpit sink saw one (fraction, "done/total") update per batch.
        assert sink.updates == [(50 / 51, "50/51"), (1.0, "51/51")]
        # The gateway's retry log rode along, so an outage mid-prefetch narrates.
        assert recorder.retry_logs == [gateway.retry_log, gateway.retry_log]


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
