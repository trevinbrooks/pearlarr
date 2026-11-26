# pyright: strict
"""A run's reads of Sonarr's episode lists.

The prefetch reads at once the series the run will check, before the check, and the check reads a series itself
when the run holds none of its rows.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import override

import pytest

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_models import SonarrEpisode, SonarrItem, SonarrSeries
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import AppConfig, Arr
from pearlarr.series_reads import SeriesReads
from pearlarr.sonarr_episodes import SonarrEpisodes

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    make_config,
    make_entry_record,
    make_run_deps,
    make_series_reads,
    sonarr_ep,
)
from .fakes import FakeSonarrClient, RecordingProgress


class _Sonarr(FakeSonarrClient):
    """Sonarr holding one episode per series, numbered by the series id, recording each read and whether it was quiet.

    A read of a series in `failing` gets no answer, as the client answers every failed request.
    """

    def __init__(self, failing: Iterable[int] = ()) -> None:
        super().__init__()
        self.failing = set(failing)
        self.reads: list[tuple[int, bool]] = []

    @property
    def series_read(self) -> list[int]:
        """The series of each read, in the order read."""

        return [series_id for series_id, _ in self.reads]

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        self.reads.append((series_id, quiet))
        return None if series_id in self.failing else [sonarr_ep(1, series_id)]


class _SlowSonarr(_Sonarr):
    """A `_Sonarr` whose reads after the first take a tenth of a second."""

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        if self.reads:
            time.sleep(0.1)
        return super().episodes(series_id, quiet=quiet)


class _TogetherSonarr(_Sonarr):
    """A `_Sonarr` whose every read waits up to `wait` seconds for a second read to be under way.

    `met` holds the series of each read that met another.
    """

    def __init__(self, wait: float) -> None:
        super().__init__()
        self._wait = wait
        self._both_started = threading.Barrier(2)
        self.met: list[int] = []

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        with contextlib.suppress(threading.BrokenBarrierError):
            self._both_started.wait(timeout=self._wait)
            self.met.append(series_id)
        return super().episodes(series_id, quiet=quiet)


class _StopOnFirstProgress:
    """A `ProgressCallback` that raises `SystemExit` as the SIGTERM handler would."""

    def progress(self, fraction: float, detail: str | None = None) -> None:
        del fraction, detail
        raise SystemExit(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """The library a run's prefetch reads, and what SeaDex, the cache and the run hold for its titles."""

    series: tuple[int, ...] = (1,)
    """The library's series, by id (each its own TVDB id)."""
    titles: Mapping[int, int] = field(default_factory=lambda: {7: 1})
    """Each AniList title the mappings know, with the series it maps to."""
    without_entry: tuple[int, ...] = ()
    """The titles SeaDex has no entry for."""
    unchanged: tuple[int, ...] = ()
    """The titles cached under their entry's current update."""
    files_changed: tuple[int, ...] = ()
    """The titles whose files changed in Sonarr since the last run."""
    unmonitored: tuple[int, ...] = ()
    """The series Sonarr doesn't monitor."""
    settings: AppConfig = field(default_factory=make_config)


def _library(run: _Run) -> list[SonarrItem]:
    """The run's library as Sonarr lists it, each series monitored unless the run says not."""

    return [SonarrSeries(id=sid, tvdbId=sid, monitored=sid not in run.unmonitored) for sid in run.series]


def _episodes(sonarr: _Sonarr, run: _Run) -> tuple[SonarrEpisodes, SeriesReads]:
    """The run's episode collaborator over `sonarr` and the reads it warms, built as a run builds them."""

    entries = {aid: make_entry_record(anilist_id=aid) for aid in run.titles}
    cache = FakeCacheStore()
    for aid in run.unchanged:
        cache.update_cached_title(Arr.SONARR, aid, {"updated_at": entries[aid].updated_at})
    deps = make_run_deps(
        config=run.settings,
        cache_store=cache,
        seadex=FakeSeaDexSource({aid: entry for aid, entry in entries.items() if aid not in run.without_entry}),
        anime_ids={f"Title {aid}": {"anilist_id": aid, "tvdb_id": sid} for aid, sid in run.titles.items()},
    )
    services = ArrRunServices(deps, Arr.SONARR)
    services.mark_dirty(run.files_changed)
    reads = make_series_reads(sonarr, library=_library(run))
    return SonarrEpisodes(deps, sonarr, services, reads), reads


class TestPrefetchEpisodes:
    """`SonarrEpisodes.prefetch` reads at once the episodes of each series the run will check.

    A series whose read fails is read again when the check asks for it, and the count and the progress take in
    every read made.
    """

    @pytest.mark.parametrize(
        ("run", "read"),
        [
            pytest.param(_Run(), [1], id="a-title-never-checked"),
            pytest.param(_Run(unchanged=(7,)), [], id="its-title-unchanged"),
            pytest.param(_Run(unchanged=(7,), files_changed=(7,)), [1], id="its-title-unchanged-but-its-files-changed"),
            pytest.param(_Run(titles={7: 1, 8: 1}, unchanged=(7,)), [1], id="one-of-its-two-titles-unchanged"),
            pytest.param(_Run(settings=make_config(ignore_anilist_ids=[7])), [], id="its-title-ignored"),
            pytest.param(_Run(without_entry=(7,)), [], id="no-seadex-entry"),
            pytest.param(
                _Run(without_entry=(7,), settings=make_config(ignore_seadex_update_times=True)),
                [],
                id="no-seadex-entry-update-times-ignored",
            ),
            pytest.param(_Run(without_entry=(7,), files_changed=(7,)), [], id="no-seadex-entry-files-changed"),
            pytest.param(
                _Run(
                    series=(1, 2),
                    titles={7: 1, 8: 2},
                    unmonitored=(1,),
                    settings=make_config(sonarr_ignore_unmonitored=True),
                ),
                [2],
                id="unmonitored-and-ignored-beside-a-monitored-series",
            ),
            pytest.param(_Run(unmonitored=(1,)), [1], id="unmonitored-not-ignored"),
        ],
    )
    def test_a_series_is_read_and_counted_only_when_one_of_its_titles_will_be_checked(
        self, run: _Run, read: list[int]
    ) -> None:
        sonarr = _Sonarr()
        episodes, _ = _episodes(sonarr, run)

        # The boot step prints this count, and "cached" when it is 0.
        warmed = episodes.prefetch(_library(run))

        assert (sonarr.series_read, warmed) == (read, len(read))

    @pytest.mark.parametrize(
        ("failing", "reads"),
        [
            pytest.param((), [(1, True)], id="answered"),
            pytest.param((1,), [(1, True), (1, False)], id="no-answer"),
        ],
    )
    def test_the_check_reads_a_series_again_only_when_the_prefetch_could_not(
        self, failing: tuple[int, ...], reads: list[tuple[int, bool]]
    ) -> None:
        run = _Run()
        sonarr = _Sonarr(failing)
        episodes, series_reads = _episodes(sonarr, run)
        episodes.prefetch(_library(run))
        sonarr.failing.clear()  # Sonarr answers again by the time the check asks

        view = series_reads.aired_series(1)

        assert isinstance(view, AiredSeries)
        assert (view.in_sonarr, sonarr.reads) == ((sonarr_ep(1, 1),), reads)

    def test_the_count_and_the_progress_take_in_every_read_made_answered_or_not(self) -> None:
        # Series 2 gets no answer, and series 3 isn't read: its title is unchanged.
        run = _Run(series=(1, 2, 3), titles={7: 1, 8: 2, 9: 3}, unchanged=(9,))
        sonarr = _Sonarr((2,))
        episodes, _ = _episodes(sonarr, run)
        progress = RecordingProgress()

        warmed = episodes.prefetch(_library(run), progress=progress)

        assert (warmed, progress.updates) == (2, [(0.5, "1/2"), (1.0, "2/2")])

    @pytest.mark.parametrize(
        ("sleep_time", "wait", "met"),
        [
            # Unthrottled the two reads meet; the long wait keeps a slow machine from failing them.
            pytest.param(0, 30.0, [1, 2], id="unthrottled"),
            # Throttled they go one at a time, so neither can meet the other however long it waits. The short wait
            # keeps the run quick. On a slow machine two reads at once can still miss each other, so this row can
            # let through a prefetch that ignores the throttle, but it never fails one that keeps it.
            pytest.param(1, 0.01, [], id="throttled"),
        ],
    )
    def test_reads_are_under_way_at_once_unless_throttled(self, sleep_time: int, wait: float, met: list[int]) -> None:
        sonarr = _TogetherSonarr(wait)
        run = _Run(series=(1, 2), titles={7: 1, 8: 2}, settings=make_config(sleep_time=sleep_time))
        episodes, _ = _episodes(sonarr, run)

        episodes.prefetch(_library(run))

        assert sorted(sonarr.met) == met

    def test_a_stop_mid_sweep_drops_the_fetches_not_yet_started(self) -> None:
        # One worker, its reads after the first a tenth of a second long, so the stop on the first read's progress
        # drops the third read before the worker is free to start it. A stop held up longer than that would see the
        # third read start and fail the test.
        sonarr = _SlowSonarr()
        run = _Run(series=(1, 2, 3), titles={7: 1, 8: 2, 9: 3}, settings=make_config(sleep_time=2))
        episodes, _ = _episodes(sonarr, run)

        with pytest.raises(SystemExit):
            episodes.prefetch(_library(run), progress=_StopOnFirstProgress())

        # The first read, plus at most the one the worker had in hand.
        assert sonarr.series_read in ([1], [1, 2])


class TestAiredSeriesRead:
    """`aired_series` fetches a cold series once per run and reports a failed fetch as `NoView.READ_FAILED`."""

    def test_second_call_serves_the_run_cache(self) -> None:
        sonarr = _Sonarr()
        reads = make_series_reads(sonarr, library=[SonarrSeries(id=1)])

        reads.aired_series(1)
        view = reads.aired_series(1)

        assert isinstance(view, AiredSeries)
        assert (view.in_sonarr, sonarr.series_read) == ((sonarr_ep(1, 1),), [1])

    def test_failed_fetch_is_no_view_and_not_cached(self) -> None:
        # A failed read is the unreadable list, distinct from a series with no episodes:
        # every verdict that depends on the episode index places nothing on it instead of reading empty.
        sonarr = _Sonarr((1,))
        reads = make_series_reads(sonarr, library=[SonarrSeries(id=1)])

        assert (reads.aired_series(1), reads.series_index(1), sonarr.series_read) == (NoView.READ_FAILED, None, [1, 1])
