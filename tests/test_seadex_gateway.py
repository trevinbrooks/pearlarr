# pyright: strict
"""Tests for SeaDexGateway bulk prefetch + by-id serving."""

import logging

import httpx
from seadex import EntryNotFoundError, EntryRecord

from seadexarr.modules.seadex_gateway import SEADEX_BATCH_SIZE, SeaDexGateway

from .builders import make_bare_instance, make_entry_record


def _rec(al_id: int) -> EntryRecord:
    return make_entry_record(anilist_id=al_id)


class FakeSeaDex:
    """Stands in for ``SeaDexEntry``: ``from_filter`` (batch) + ``from_id`` (single)."""

    def __init__(self, entries: dict[int, EntryRecord], *, fail_filter: bool = False) -> None:
        self.entries = entries
        self.fail_filter = fail_filter
        self.filter_calls: list[str] = []
        self.from_id_calls: list[int] = []

    def from_filter(self, filter_str: str) -> list[EntryRecord]:
        self.filter_calls.append(filter_str)
        if self.fail_filter:
            raise httpx.ConnectError("down")
        ids = [int(part.split("=")[1]) for part in filter_str.split("||")]
        return [self.entries[i] for i in ids if i in self.entries]

    def from_id(self, al_id: int) -> EntryRecord:
        self.from_id_calls.append(al_id)
        if al_id in self.entries:
            return self.entries[al_id]
        raise EntryNotFoundError("nope")


class _Recorder:
    """A ``PrefetchProgress`` sink that records every ``progress`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.calls.append((fraction, detail))


def _gateway(fake: FakeSeaDex) -> SeaDexGateway:
    logger = logging.getLogger("test_seadex_gateway")
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    # Bypass __init__ (which would build a real SeaDexEntry client) and inject the
    # fake + the per-run cache fields directly.
    return make_bare_instance(
        SeaDexGateway,
        logger=logger,
        seadex=fake,
        _entry_cache={},
        _prefetched=set(),
    )


class TestSeaDexPrefetch:
    def test_prefetch_serves_from_cache_without_from_id(self) -> None:
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2), 3: _rec(3)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2, 3])
        assert len(fake.filter_calls) == 1  # one batch
        assert gateway.entry(1) is fake.entries[1]
        assert gateway.entry(2) is fake.entries[2]
        assert fake.from_id_calls == []  # never fell back to per-id

    def test_batch_emits_or_ed_alid_filter(self) -> None:
        # Drift guard: pins the exact OR-filter syntax prefetch emits - the format
        # the real SeaDex server expects and FakeSeaDex.from_filter parses. The
        # round-trip can't catch a key-name co-drift (the fake's parser ignores the
        # clause key), so assert the string itself.
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2), 3: _rec(3)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2, 3])
        assert fake.filter_calls == ["alID=1 || alID=2 || alID=3"]

    def test_prefetched_absent_id_returns_none_without_fallback(self) -> None:
        fake = FakeSeaDex({1: _rec(1)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2])  # 2 has no SeaDex entry
        assert gateway.entry(2) is None
        assert fake.from_id_calls == []  # known-absent -> no fallback call

    def test_second_prefetch_skips_known_absent_id(self) -> None:
        fake = FakeSeaDex({1: _rec(1)})  # id 2 has no SeaDex entry
        gateway = _gateway(fake)
        gateway.prefetch([1, 2])
        n = len(fake.filter_calls)
        gateway.prefetch([1, 2])  # both known: 1 cached, 2 prefetched-absent
        assert len(fake.filter_calls) == n  # no re-fetch of known ids
        assert gateway.entry(2) is None
        assert fake.from_id_calls == []  # never fell back to per-id

    def test_unprefetched_id_falls_back_to_from_id(self) -> None:
        fake = FakeSeaDex({9: _rec(9)})
        gateway = _gateway(fake)
        assert gateway.entry(9) is fake.entries[9]
        assert fake.from_id_calls == [9]

    def test_unprefetched_missing_id_returns_none(self) -> None:
        fake = FakeSeaDex({})
        gateway = _gateway(fake)
        assert gateway.entry(123) is None  # from_id raises EntryNotFound -> None
        assert fake.from_id_calls == [123]

    def test_batch_outage_falls_back_per_id(self) -> None:
        fake = FakeSeaDex({1: _rec(1)}, fail_filter=True)
        gateway = _gateway(fake)
        gateway.prefetch([1])  # the batch raises -> chunk left unprefetched
        assert gateway.entry(1) is fake.entries[1]  # fell back to from_id
        assert fake.from_id_calls == [1]

    def test_batches_respect_batch_size(self) -> None:
        ids = list(range(1, SEADEX_BATCH_SIZE * 2 + 6))  # two full batches + a partial
        fake = FakeSeaDex({i: _rec(i) for i in ids})
        gateway = _gateway(fake)
        gateway.prefetch(ids)
        assert len(fake.filter_calls) == 3

    def test_prefetch_returns_missing_count(self) -> None:
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2)})
        gateway = _gateway(fake)
        assert gateway.prefetch([1, 2]) == 2

    def test_warm_prefetch_returns_zero_and_skips_sink(self) -> None:
        fake = FakeSeaDex({1: _rec(1)})
        gateway = _gateway(fake)
        gateway.prefetch([1])  # warm the per-run cache first
        rec = _Recorder()
        assert gateway.prefetch([1], progress=rec) == 0  # nothing left to fetch
        assert rec.calls == []  # no work -> no progress drive

    def test_prefetch_drives_progress_per_batch(self) -> None:
        ids = list(range(1, SEADEX_BATCH_SIZE * 2 + 6))  # three batches
        fake = FakeSeaDex({i: _rec(i) for i in ids})
        gateway = _gateway(fake)
        rec = _Recorder()
        assert gateway.prefetch(ids, progress=rec) == len(ids)
        assert len(rec.calls) == 3  # one drive per batch
        assert rec.calls[-1] == (1.0, f"{len(ids)}/{len(ids)}")  # ends complete
