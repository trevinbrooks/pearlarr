# pyright: strict
"""The scene mapping cache: which series are read again, what a read stores, and what is served."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pearlarr.cache import CacheStore
from pearlarr.scene_numbers import CachedSceneMappings, SceneMapping
from pearlarr.stamps import parse_stamp
from pearlarr.tvdb_orders import AIRED_ORDER, SonarrSeriesOrders
from pearlarr.xem_gateway import XemGateway

from .builders import FakeCacheStore, stored_scene_mappings
from .fakes import FakeXemClient

_SERIES = 79214
_OTHER = 257756
_MAPPINGS = (SceneMapping(0, 1, 2, 1, 1), SceneMapping(1, 1, 1, 1, 1))
_OLDER = (SceneMapping(1, 1, 1, 1, 1),)
"""What an earlier read stored."""


def _in_another_order(*tvdb_ids: int) -> list[SonarrSeriesOrders]:
    return [SonarrSeriesOrders(tvdb_id, "dvd", None) for tvdb_id in tvdb_ids]


def _gateway(client: FakeXemClient, stored: Mapping[int, CachedSceneMappings] | None = None) -> XemGateway:
    cache = FakeCacheStore()
    for tvdb_id, record in (stored or {}).items():
        cache.put_scene_mappings(tvdb_id, record)
    return XemGateway(client=client, cache_store=cache)


class _RecordingProgress:
    """Captures each (fraction, detail) update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.updates.append((fraction, detail))


def test_a_series_never_read_is_read_and_its_mappings_served() -> None:
    gateway = _gateway(FakeXemClient({_SERIES: _MAPPINGS}))

    assert gateway.scene_mappings(_SERIES) is None
    assert gateway.prefetch(_in_another_order(_SERIES), dry_run=True) == 1

    assert gateway.scene_mappings(_SERIES) == _MAPPINGS


@pytest.mark.parametrize("season_type", [AIRED_ORDER, ""])
def test_a_series_in_aired_order_is_never_asked_for(season_type: str) -> None:
    client = FakeXemClient({_SERIES: _MAPPINGS})
    gateway = _gateway(client)

    assert gateway.prefetch([SonarrSeriesOrders(_SERIES, season_type, None)], dry_run=True) == 0

    assert client.asked == []
    assert gateway.scene_mappings(_SERIES) is None


@pytest.mark.parametrize(
    ("fresh_for", "asked"),
    [
        pytest.param(timedelta(days=-1), [_SERIES], id="past-fresh-until"),
        pytest.param(timedelta(days=1), [], id="fresh"),
    ],
)
def test_stored_mappings_are_read_again_once_their_fresh_until_passes(fresh_for: timedelta, asked: list[int]) -> None:
    client = FakeXemClient({_SERIES: _MAPPINGS})
    gateway = _gateway(client, {_SERIES: stored_scene_mappings(_OLDER, fresh_for=fresh_for)})

    gateway.prefetch(_in_another_order(_SERIES), dry_run=True)

    assert client.asked == asked
    assert gateway.scene_mappings(_SERIES) == (_MAPPINGS if asked else _OLDER)


def test_a_read_keeps_the_mappings_fresh_for_a_week() -> None:
    cache = FakeCacheStore()
    gateway = XemGateway(client=FakeXemClient({_SERIES: _MAPPINGS}), cache_store=cache)
    before = datetime.now().replace(microsecond=0)

    gateway.prefetch(_in_another_order(_SERIES), dry_run=True)

    stored = cache.get_scene_mappings(_SERIES)
    assert stored is not None
    assert before + timedelta(days=7) <= parse_stamp(stored.fresh_until) <= datetime.now() + timedelta(days=7)


def test_a_series_thexem_has_no_mappings_for_is_read_and_served_as_having_none() -> None:
    client = FakeXemClient()
    gateway = _gateway(client)

    assert [gateway.prefetch(_in_another_order(_SERIES), dry_run=True) for _ in range(2)] == [1, 0]

    assert gateway.scene_mappings(_SERIES) == ()
    assert client.asked == [_SERIES]


def test_an_answer_that_cannot_be_used_keeps_the_stored_mappings_and_their_stamp() -> None:
    cache = FakeCacheStore()
    stale = stored_scene_mappings(_OLDER, fresh_for=timedelta(days=-1))
    cache.put_scene_mappings(_SERIES, stale)
    client = FakeXemClient(unusable=frozenset({_SERIES}))
    gateway = XemGateway(client=client, cache_store=cache)

    assert [gateway.prefetch(_in_another_order(_SERIES), dry_run=True) for _ in range(2)] == [0, 0]

    assert cache.get_scene_mappings(_SERIES) == stale
    assert gateway.scene_mappings(_SERIES) == _OLDER
    # Still due, so each run asks again.
    assert client.asked == [_SERIES, _SERIES]


def test_an_answer_that_cannot_be_used_leaves_a_series_never_read_and_the_next_one_is_still_read() -> None:
    client = FakeXemClient({_OTHER: _MAPPINGS}, unusable=frozenset({_SERIES}))
    gateway = _gateway(client)

    assert gateway.prefetch(_in_another_order(_SERIES, _OTHER), dry_run=True) == 1

    assert gateway.scene_mappings(_SERIES) is None
    assert gateway.scene_mappings(_OTHER) == _MAPPINGS
    assert gateway.unreachable is False


def test_an_unreachable_thexem_keeps_what_is_stored_and_leaves_the_rest_unasked() -> None:
    cache = FakeCacheStore()
    stale = stored_scene_mappings(_OLDER, fresh_for=timedelta(days=-1))
    cache.put_scene_mappings(_SERIES, stale)
    client = FakeXemClient({_OTHER: _MAPPINGS}, failing=frozenset({_SERIES}))
    gateway = XemGateway(client=client, cache_store=cache)

    assert gateway.prefetch(_in_another_order(_SERIES, _OTHER), dry_run=True) == 0

    assert gateway.unreachable is True
    assert client.asked == [_SERIES]
    assert cache.get_scene_mappings(_SERIES) == stale
    assert gateway.scene_mappings(_SERIES) == _OLDER
    assert gateway.scene_mappings(_OTHER) is None


def test_progress_counts_each_series_asked_for_one_whose_answer_cannot_be_used_too() -> None:
    progress = _RecordingProgress()
    # The second of three due series can't be used. A fourth is fresh and a fifth in aired order: neither counts.
    client = FakeXemClient(unusable=frozenset({2}))
    gateway = _gateway(client, {4: stored_scene_mappings(_OLDER, fresh_for=timedelta(days=1))})
    series = [*_in_another_order(1, 2, 3, 4), SonarrSeriesOrders(5, AIRED_ORDER, None)]

    gateway.prefetch(series, dry_run=True, progress=progress)

    assert progress.updates == [(1 / 3, "1/3"), (2 / 3, "2/3"), (1.0, "3/3")]


@pytest.mark.parametrize(
    ("dry_run", "kept"), [pytest.param(False, True, id="real"), pytest.param(True, False, id="preview")]
)
def test_each_series_mappings_are_saved_as_written_unless_in_a_dry_run(
    tmp_path: Path, dry_run: bool, kept: bool
) -> None:
    path = str(tmp_path / "cache.db")
    store = CacheStore.load(path, config_hash="x")
    XemGateway(client=FakeXemClient({_SERIES: _MAPPINGS}), cache_store=store).prefetch(
        _in_another_order(_SERIES), dry_run=dry_run
    )
    # Closing without the end-of-run save is how an interrupted run ends.
    store.close()

    reopened = CacheStore.load(path, config_hash="x")
    assert (reopened.get_scene_mappings(_SERIES) is not None) is kept
    reopened.close()
