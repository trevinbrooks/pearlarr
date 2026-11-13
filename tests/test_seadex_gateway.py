# pyright: strict
"""Tests for SeaDexGateway bulk prefetch + by-id serving."""

import httpx
import pytest
from seadex import EntryNotFoundError, EntryRecord

from pearlarr.output import Severity
from pearlarr.seadex_gateway import SEADEX_BATCH_SIZE, SeaDexGateway, SeaDexNoEntry

from .builders import make_entry_record
from .fakes import diagnostic_messages, install_recording_hub


def _rec(al_id: int) -> EntryRecord:
    return make_entry_record(anilist_id=al_id)


class FakeSeaDex:
    """Stands in for `SeaDexEntry`: `from_filter` (batch) + `from_id` (single).

    `filter_error` / `from_id_error` fail every call with that error (SeaDex
    fully unreachable). `filter_blips` / `from_id_blips` fail just the first N calls then
    recover (a transient blip the retry absorbs).
    """

    def __init__(
        self,
        entries: dict[int, EntryRecord],
        *,
        filter_error: httpx.HTTPError | None = None,
        filter_blips: int = 0,
        from_id_error: httpx.HTTPError | None = None,
        from_id_blips: int = 0,
    ) -> None:
        self.entries = entries
        self.filter_error = filter_error
        self.filter_blips = filter_blips
        self.from_id_error = from_id_error
        self.from_id_blips = from_id_blips
        self.filter_calls: list[str] = []
        self.from_id_calls: list[int] = []

    def from_filter(self, filter_str: str) -> list[EntryRecord]:
        self.filter_calls.append(filter_str)
        if self.filter_error is not None:
            raise self.filter_error
        if self.filter_blips > 0:
            self.filter_blips -= 1
            raise httpx.ConnectError("blip")
        ids = [int(part.split("=")[1]) for part in filter_str.split("||")]
        return [self.entries[i] for i in ids if i in self.entries]

    def from_id(self, al_id: int) -> EntryRecord:
        self.from_id_calls.append(al_id)
        if self.from_id_error is not None:
            raise self.from_id_error
        if self.from_id_blips > 0:
            self.from_id_blips -= 1
            raise httpx.ConnectError("blip")
        if al_id in self.entries:
            return self.entries[al_id]
        raise EntryNotFoundError("nope")


class _Recorder:
    """A `ProgressCallback` that records every `progress` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.calls.append((fraction, detail))


def _gateway(fake: FakeSeaDex) -> SeaDexGateway:
    # The fake satisfies SeaDexEntryApi structurally, so the real ctor applies.
    return SeaDexGateway(client=fake)


def _status_error(status: int, location: str | None = None) -> httpx.HTTPStatusError:
    """An `HTTPStatusError` for a SeaDex records query, redirecting to `location` if given."""

    url = "https://releases.moe/api/collections/entries/records"
    request = httpx.Request("GET", url)
    response = httpx.Response(status, request=request)
    if location is not None:
        response.next_request = httpx.Request("GET", location)
    return httpx.HTTPStatusError(f"error '{status}' for url '{url}'", request=request, response=response)


class TestSeaDexPrefetch:
    """`prefetch` batches entries so `entry` serves cache hits and known-absent ids without a per-id call.

    Only unprefetched ids go to `from_id`, and a single failed batch
    chunk retries silently before SeaDex is marked unreachable.
    """

    def test_prefetch_serves_from_cache_without_from_id(self) -> None:
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2), 3: _rec(3)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2, 3])
        assert len(fake.filter_calls) == 1  # one batch
        assert gateway.entry(1) is fake.entries[1]
        assert gateway.entry(2) is fake.entries[2]
        assert fake.from_id_calls == []  # never called per-id

    def test_batch_emits_or_ed_alid_filter(self) -> None:
        # Drift guard: pins the exact OR-filter syntax prefetch emits - the format
        # the real SeaDex server expects and FakeSeaDex.from_filter parses. The
        # round-trip can't catch a key-name co-drift (the fake's parser ignores the
        # clause key), so assert the string itself.
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2), 3: _rec(3)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2, 3])
        assert fake.filter_calls == ["alID=1 || alID=2 || alID=3"]

    def test_prefetched_absent_id_is_no_entry_without_a_per_id_call(self) -> None:
        fake = FakeSeaDex({1: _rec(1)})
        gateway = _gateway(fake)
        gateway.prefetch([1, 2])  # 2 has no SeaDex entry
        assert gateway.entry(2) is SeaDexNoEntry.NOT_FOUND
        assert fake.from_id_calls == []  # known-absent -> no per-id call

    def test_second_prefetch_skips_known_absent_id(self) -> None:
        fake = FakeSeaDex({1: _rec(1)})  # id 2 has no SeaDex entry
        gateway = _gateway(fake)
        gateway.prefetch([1, 2])
        n = len(fake.filter_calls)
        gateway.prefetch([1, 2])  # both known: 1 cached, 2 prefetched-absent
        assert len(fake.filter_calls) == n  # no re-fetch of known ids
        assert gateway.entry(2) is SeaDexNoEntry.NOT_FOUND
        assert fake.from_id_calls == []  # never called per-id

    def test_unprefetched_id_is_looked_up_by_from_id(self) -> None:
        fake = FakeSeaDex({9: _rec(9)})
        gateway = _gateway(fake)
        assert gateway.entry(9) is fake.entries[9]
        assert fake.from_id_calls == [9]

    def test_unprefetched_absent_id_is_no_entry(self) -> None:
        fake = FakeSeaDex({})
        gateway = _gateway(fake)
        assert gateway.entry(123) is SeaDexNoEntry.NOT_FOUND  # from_id raises EntryNotFound
        assert fake.from_id_calls == [123]

    def test_transient_batch_blip_recovers_on_retry(self) -> None:
        # A single failed batch (one 502 among many) is retried immediately and
        # silently: the run keeps its SeaDex lookups, SeaDex stays reachable, no warning.
        ids = list(range(1, SEADEX_BATCH_SIZE + 6))  # two batches
        fake = FakeSeaDex({i: _rec(i) for i in ids}, filter_blips=1)
        gateway = _gateway(fake)
        recording = install_recording_hub()

        gateway.prefetch(ids)

        assert len(fake.filter_calls) == 3  # chunk 1 retried once, chunk 2 clean
        assert gateway.unreachable is False
        assert gateway.entry(ids[0]) is fake.entries[ids[0]]
        assert gateway.entry(ids[-1]) is fake.entries[ids[-1]]
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_batch_unreachable_warns_once_and_short_circuits(self) -> None:
        # A chunk failing TWICE (retry exhausted) marks SeaDex unreachable: it warns
        # once, later batches never hit the network, and every entry() degrades
        # straight to an UNREACHABLE `SeaDexNoEntry` with zero from_id calls.
        ids = list(range(1, SEADEX_BATCH_SIZE * 2 + 6))  # three batches
        fake = FakeSeaDex({i: _rec(i) for i in ids}, filter_error=httpx.ConnectError("down"))
        gateway = _gateway(fake)
        recording = install_recording_hub()

        gateway.prefetch(ids)

        assert len(fake.filter_calls) == 2  # chunk 1 + its retry, batches 2+3 short-circuited
        assert gateway.unreachable is True
        warnings = diagnostic_messages(recording, Severity.WARNING)
        assert len(warnings) == 1
        assert "SeaDex request failed (ConnectError)" in warnings[0]
        # The per-id call is muted too: no fresh timeout per title.
        assert gateway.entry(ids[0]) is SeaDexNoEntry.UNREACHABLE
        assert fake.from_id_calls == []
        assert len(diagnostic_messages(recording, Severity.WARNING)) == 1  # still just the one

    @pytest.mark.parametrize(
        ("error", "cause"),
        [
            (
                _status_error(301, "https://seadex.moe/api/collections/entries/records"),
                "HTTP 301 redirect to seadex.moe",
            ),
            (_status_error(503), "HTTP 503"),
        ],
    )
    def test_unreachable_warning_names_the_failure(self, error: httpx.HTTPError, cause: str) -> None:
        fake = FakeSeaDex({1: _rec(1)}, filter_error=error)
        recording = install_recording_hub()

        _gateway(fake).prefetch([1])

        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert f"SeaDex request failed ({cause})" in warning

    def test_unreachable_prefetch_still_drives_progress_to_completion(self) -> None:
        # The boot step sink must not hang when SeaDex turns unreachable mid-prefetch: every chunk
        # still reports, ending at 1.0.
        ids = list(range(1, SEADEX_BATCH_SIZE + 6))  # two batches
        fake = FakeSeaDex({}, filter_error=httpx.ConnectError("down"))
        gateway = _gateway(fake)
        rec = _Recorder()

        assert gateway.prefetch(ids, progress=rec) == len(ids)
        assert rec.calls[-1] == (1.0, f"{len(ids)}/{len(ids)}")

    def test_failed_chunk_ids_are_not_marked_absent(self) -> None:
        # An id in a failed chunk is transiently unread, never a remembered
        # "no entry": a later prefetch still counts it as work to do (it
        # stays out of the prefetched set) rather than returning 0.
        fake = FakeSeaDex({1: _rec(1)}, filter_error=httpx.ConnectError("down"))
        gateway = _gateway(fake)
        gateway.prefetch([1])
        assert gateway.prefetch([1]) == 1

    def test_single_lookup_timeout_marks_seadex_unreachable_and_warns_once(self) -> None:
        # The module contract: an unreachable SeaDex degrades to an UNREACHABLE `SeaDexNoEntry`. A
        # ReadTimeout on the per-id call (httpx.HTTPError, not just
        # ConnectError) must not unwind the run - and a SECOND lookup
        # short-circuits without another network attempt or warning.
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2)}, from_id_error=httpx.ReadTimeout("slow"))
        gateway = _gateway(fake)
        recording = install_recording_hub()

        assert gateway.unreachable is False
        assert gateway.entry(1) is SeaDexNoEntry.UNREACHABLE
        assert gateway.entry(2) is SeaDexNoEntry.UNREACHABLE
        assert gateway.unreachable is True
        assert fake.from_id_calls == [1, 1]  # the retry, then id 2 never hit the network
        assert len(diagnostic_messages(recording, Severity.WARNING)) == 1

    def test_single_lookup_blip_recovers_on_retry(self) -> None:
        # A lone transient blip on the per-id call is retried immediately
        # and silently, mirroring the batch path: the entry is served, SeaDex
        # stays reachable, no warning.
        fake = FakeSeaDex({1: _rec(1)}, from_id_blips=1)
        gateway = _gateway(fake)
        recording = install_recording_hub()

        assert gateway.entry(1) is fake.entries[1]
        assert gateway.unreachable is False
        assert fake.from_id_calls == [1, 1]  # the blip + its retry
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_single_lookup_double_blip_marks_seadex_unreachable(self) -> None:
        # The retry absorbs exactly one blip: a lookup failing twice flips the
        # run-wide unreachable flag with the single batch-path-style warning.
        fake = FakeSeaDex({1: _rec(1)}, from_id_blips=2)
        gateway = _gateway(fake)
        recording = install_recording_hub()

        assert gateway.entry(1) is SeaDexNoEntry.UNREACHABLE
        assert gateway.unreachable is True
        assert fake.from_id_calls == [1, 1]
        warnings = diagnostic_messages(recording, Severity.WARNING)
        assert len(warnings) == 1
        assert "SeaDex request failed (ConnectError)" in warnings[0]

    def test_batches_respect_batch_size(self) -> None:
        ids = list(range(1, SEADEX_BATCH_SIZE * 2 + 6))  # two full batches + a partial
        fake = FakeSeaDex({i: _rec(i) for i in ids})
        gateway = _gateway(fake)
        gateway.prefetch(ids)
        assert len(fake.filter_calls) == 3

    def test_prefetch_returns_the_count_to_fetch(self) -> None:
        fake = FakeSeaDex({1: _rec(1), 2: _rec(2)})
        gateway = _gateway(fake)
        assert gateway.prefetch([1, 2]) == 2

    def test_warm_prefetch_returns_zero_and_skips_progress(self) -> None:
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
