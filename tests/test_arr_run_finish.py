# pyright: strict
"""Tests for the end of an arr run: what it shows, counts, saves and sends once the scan is done.

Each test runs a whole arr run, the loop and its services built by their constructors, over a cache an earlier run
left pending imports in. qBittorrent is scripted, time is a fake clock, and the arr's side of an import is a
scripted strategy. The outcome is read off the events the run showed, the cache once it is closed and opened again,
what the arr was asked to do, and what the wait webhook was sent. That a run whose title or library item failed
still ends with its summary is pinned in `test_arr_run_scan`, and that a failed push only warns in `test_run_finalize`.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast, override

import pytest
import respx

from pearlarr.boundary import Json
from pearlarr.cache import CacheStore
from pearlarr.config import AppConfig, Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import ArrRunFinished, ArrRunSummaryReady, ScanFinished, WaitStarted
from pearlarr.pending_import import PendingImport
from pearlarr.wait_types import ImportProgress, ImportWaitMode

from .builders import attempt_result, drive_arr_run, make_config, make_run_deps, pending_import
from .fakes import FakeArrItem, FakeClock, RecordingHub, install_recording_hub
from .wait_fakes import TORRENT_H_COMPLETE, FakeQbit, FakeTorrent, RecordingStrategy

_ALL_IMPORTED = (attempt_result(),)
"""The arr's answer to every import it is asked for: all of the torrent's files are in."""

_ALL_IN = (ImportProgress(1, 1, determinate=True),)
"""The arr's answer to every import progress read: all of the torrent's files are in."""

_FINISHED = FakeTorrent(is_complete=True, content_path="/downloads/done")
"""A torrent qBittorrent has finished downloading."""

_DOWNLOADING = FakeTorrent(progress=0.5)
"""A torrent qBittorrent is still downloading."""

_WEBHOOK = "https://hook.example/wait"
"""The wait webhook the notifying tests set in the config."""


def _pending(infohash: str) -> PendingImport:
    """A pending import of Frieren on series 7 by SubGroup, so its wait row reads `Frieren · SubGroup`."""

    return pending_import(infohash=infohash, title="Frieren", series_id=7, release_group="SubGroup")


def _cache_left_with(cache: Path, arr: Arr, *pending: PendingImport) -> None:
    """Save `pending` for `arr` into the cache file at `cache`, as an earlier run would have left them."""

    store = CacheStore.load(str(cache), config_hash="")
    for each in pending:
        store.put_pending_import_row(arr, each.infohash, each.to_json())
    store.save(dry_run=False)
    store.close()


def _saved_pending_imports(cache: Path, arr: Arr) -> set[str]:
    """The hashes of the pending imports the cache file at `cache` holds for `arr`, read fresh."""

    store = CacheStore.load(str(cache), config_hash="")
    try:
        return set(store.get_pending_import_rows(arr))
    finally:
        store.close()


def _run(
    cache: Path,
    strategy: RecordingStrategy,
    qbit: FakeQbit | None,
    *,
    mode: ImportWaitMode,
    dry_run: bool = False,
    arr: Arr = Arr.SONARR,
    config: AppConfig | None = None,
) -> None:
    """One whole arr run of `strategy` over the cache file at `cache`, closed after it as bootstrap closes it.

    The deps, the loop and its services are built the way a run builds them. `mode` goes to `run_arr` as the
    `--import-wait-mode` override does, so it wins over the config's `imports.wait_mode`, and `dry_run` as
    `--dry-run` does. `qbit` None is a run with no qBittorrent configured. Every sleep passes one poll interval on a
    fake clock, so no wait sleeps for real.
    """

    store = CacheStore.load(str(cache), config_hash="")
    try:
        client = None if qbit is None else qbit.client()
        deps = make_run_deps(config=config, cache_store=store, qbit=client, arr=arr, clock=FakeClock(step=30))
        drive_arr_run(deps, lambda _: strategy, arr=arr, dry_run=dry_run, import_wait_mode=mode)
    finally:
        store.close()


def _finish_steps(recording: RecordingHub) -> list[str]:
    """What the end of the run showed, in order: the scan closing, each wait opening, the summary, the run closing."""

    steps: list[str] = []
    for event in recording.events:
        match event:
            case ScanFinished(arr=arr):
                steps.append(f"{arr} scan finished")
            case WaitStarted(kind=kind):
                steps.append(f"{kind.name.lower()} wait")
            case ArrRunSummaryReady():
                steps.append("summary")
            case ArrRunFinished(arr=arr):
                steps.append(f"{arr} run finished")
            case _:
                pass
    return steps


class TestWhatARunShows:
    """The end of a run closes the scan first, shows the summary before the blocking wait, and closes the run last.

    With the scan closed first, what the waits report prints at run level, never inside the last title. The summary
    comes before the blocking wait, so the run's tally shows while it waits. Closing the run is its last event.
    """

    @pytest.mark.parametrize(
        ("mode", "steps"),
        [
            pytest.param(
                ImportWaitMode.BLOCKING,
                ["sonarr scan finished", "summary", "blocking wait", "sonarr run finished"],
                id="blocking",
            ),
            pytest.param(
                ImportWaitMode.DEFERRED,
                ["sonarr scan finished", "deferred wait", "summary", "sonarr run finished"],
                id="deferred",
            ),
        ],
    )
    def test_a_sonarr_run_shows_its_wait_in_order(self, tmp_path: Path, mode: ImportWaitMode, steps: list[str]) -> None:
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("h"))
        recording = install_recording_hub()

        _run(cache, RecordingStrategy(), FakeQbit(TORRENT_H_COMPLETE), mode=mode)

        assert _finish_steps(recording) == steps

    def test_a_radarr_run_looks_at_its_imports_before_the_summary_in_blocking_mode(self, tmp_path: Path) -> None:
        # Radarr has no blocking wait, so the deferred wait is the only look its pending imports get. It stores a
        # movie's pending import on series 0, with no episode map or SeaDex file names.
        cache = tmp_path / "cache.db"
        movie = pending_import(infohash="h", series_id=0, file_episode_map={}, seadex_files=())
        _cache_left_with(cache, Arr.RADARR, movie)
        recording = install_recording_hub()

        _run(
            cache,
            RecordingStrategy(supports_blocking_wait=False),
            FakeQbit(TORRENT_H_COMPLETE),
            mode=ImportWaitMode.BLOCKING,
            arr=Arr.RADARR,
        )

        assert _finish_steps(recording) == ["radarr scan finished", "deferred wait", "summary", "radarr run finished"]

    @pytest.mark.parametrize(
        ("stored", "mode", "dry_run"),
        [
            # A dry run imports nothing, so it looks at no pending import whatever the mode.
            pytest.param((_pending("h"),), ImportWaitMode.BLOCKING, True, id="dry-run"),
            pytest.param((_pending("h"),), ImportWaitMode.OFF, False, id="wait-off"),
            pytest.param((), ImportWaitMode.BLOCKING, False, id="nothing-pending-blocking"),
            pytest.param((), ImportWaitMode.DEFERRED, False, id="nothing-pending-deferred"),
        ],
    )
    def test_a_run_with_no_wait_to_run_shows_none(
        self, tmp_path: Path, stored: tuple[PendingImport, ...], mode: ImportWaitMode, dry_run: bool
    ) -> None:
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, *stored)
        recording = install_recording_hub()

        _run(cache, RecordingStrategy(), FakeQbit(TORRENT_H_COMPLETE), mode=mode, dry_run=dry_run)

        assert _finish_steps(recording) == ["sonarr scan finished", "summary", "sonarr run finished"]


class _SeriesInLibrary(RecordingStrategy):
    """Sonarr's side of a run over a library of series 7 under one title, so the scan reports its pending imports."""

    @override
    def get_items(self) -> list[FakeArrItem]:
        return [FakeArrItem(item_id=7)]

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {1: TitleMapping(anilist_id=1)}

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        return item.id


class TestWhatARunCounts:
    """The summary counts each pending import where the run left it, and says when the run could grab nothing."""

    def test_the_summary_counts_each_carried_over_import_where_it_ended(self, tmp_path: Path) -> None:
        # The arr answers every import and progress read the same (every file in), so qBittorrent decides each end:
        # "a" is finished when the scan reports on it, so the scan shows it imported; "b" is downloading then and
        # finished by the deferred wait, which imports it; "c" is still downloading at the end.
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("a"), _pending("b"), _pending("c"))
        qbit = FakeQbit({"a": [_FINISHED], "b": [_DOWNLOADING, _FINISHED], "c": [_DOWNLOADING]})
        strategy = _SeriesInLibrary(attempts=_ALL_IMPORTED, progress=_ALL_IN)
        recording = install_recording_hub()

        _run(cache, strategy, qbit, mode=ImportWaitMode.DEFERRED)

        (summary,) = recording.of_type(ArrRunSummaryReady)
        assert summary.summary.tally.imported == 2
        assert summary.summary.tally.downloading == 1
        assert summary.summary.tally.downloaded == 0

    @pytest.mark.parametrize(
        ("qbit", "dry_run", "note"),
        [
            pytest.param(FakeQbit(), False, None, id="real-run"),
            pytest.param(FakeQbit(), True, "nothing grabbed", id="dry-run"),
            pytest.param(None, False, "qBittorrent not configured; nothing grabbed", id="no-qbittorrent"),
        ],
    )
    def test_the_summary_notes_a_dry_run_and_why(
        self, tmp_path: Path, qbit: FakeQbit | None, dry_run: bool, note: str | None
    ) -> None:
        recording = install_recording_hub()

        _run(tmp_path / "cache.db", RecordingStrategy(), qbit, mode=ImportWaitMode.OFF, dry_run=dry_run)

        (summary,) = recording.of_type(ArrRunSummaryReady)
        assert summary.summary.dry_run_note == note


class TestWhatARunSaves:
    """What a run leaves in the cache and the arr for the next run, a stop in its finish included.

    The save trails the blocking wait, so it holds the imports the wait finished. It runs in a finally around both
    waits, so a stop never rolls back what the run did when the cache is closed. A stop is the exit a SIGTERM
    raises, landing here while qBittorrent is read. Closing the run is not shown on the way out: bootstrap's unwind
    shows it, so it lands once.
    """

    def test_the_saved_cache_drops_the_import_the_blocking_wait_finished(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("h"))

        _run(
            cache, RecordingStrategy(attempts=_ALL_IMPORTED), FakeQbit(TORRENT_H_COMPLETE), mode=ImportWaitMode.BLOCKING
        )

        assert _saved_pending_imports(cache, Arr.SONARR) == set()

    @pytest.mark.parametrize("mode", [ImportWaitMode.BLOCKING, ImportWaitMode.DEFERRED])
    def test_a_run_finishes_the_cleanup_an_earlier_run_left_owed(self, tmp_path: Path, mode: ImportWaitMode) -> None:
        # An earlier run imported "h" but did not finish its cleanup, so it is stored owing it. The wait leaves
        # such an import out of its rows: the finish is the only step that finishes it.
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, replace(_pending("h"), awaiting_cleanup=True))
        strategy = RecordingStrategy(attempts=_ALL_IMPORTED)

        _run(cache, strategy, FakeQbit(TORRENT_H_COMPLETE), mode=mode)

        assert strategy.removal_calls == ["h"]
        assert _saved_pending_imports(cache, Arr.SONARR) == set()

    def test_a_run_with_the_wait_off_leaves_its_pending_imports_alone(self, tmp_path: Path) -> None:
        # Series 7 is in the library and the arr has all of "h" in, so the scan's look at "h" would finish it.
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("h"))
        strategy = _SeriesInLibrary(progress=_ALL_IN)

        _run(cache, strategy, FakeQbit(TORRENT_H_COMPLETE), mode=ImportWaitMode.OFF)

        assert strategy.import_calls == []
        assert _saved_pending_imports(cache, Arr.SONARR) == {"h"}

    def test_a_stop_in_the_deferred_wait_still_saves_the_import_it_finished(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("a"), _pending("b"))
        # The stop lands while qBittorrent is read for "b", after "a" was imported.
        qbit = FakeQbit({"a": [_FINISHED], "b": [SystemExit(0)]})
        recording = install_recording_hub()

        with pytest.raises(SystemExit):
            _run(cache, RecordingStrategy(attempts=_ALL_IMPORTED), qbit, mode=ImportWaitMode.DEFERRED)

        assert _saved_pending_imports(cache, Arr.SONARR) == {"b"}
        assert _finish_steps(recording) == ["sonarr scan finished", "deferred wait"]

    def test_a_stop_in_the_blocking_wait_still_saves_the_import_it_finished(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("a"), _pending("b"))
        # "b" is still downloading at the first poll, and the stop lands while qBittorrent is read on the next.
        qbit = FakeQbit({"a": [_FINISHED], "b": [_DOWNLOADING, SystemExit(0)]})
        recording = install_recording_hub()

        with pytest.raises(SystemExit):
            _run(cache, RecordingStrategy(attempts=_ALL_IMPORTED), qbit, mode=ImportWaitMode.BLOCKING)

        assert _saved_pending_imports(cache, Arr.SONARR) == {"b"}
        assert _finish_steps(recording) == ["sonarr scan finished", "summary", "blocking wait"]


class TestWhatARunSends:
    """A blocking wait that finished an import sends its report to the wait webhook, unless notifying is off."""

    @respx.mock
    def test_a_blocking_wait_that_finished_an_import_sends_the_wait_webhook(self, tmp_path: Path) -> None:
        hook = respx.post(_WEBHOOK).respond()
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("h"))
        config = make_config(url="http://sonarr", api_key="key", wait_webhook_url=_WEBHOOK)

        _run(
            cache,
            RecordingStrategy(attempts=_ALL_IMPORTED),
            FakeQbit(TORRENT_H_COMPLETE),
            mode=ImportWaitMode.BLOCKING,
            config=config,
        )

        assert hook.call_count == 1
        report = cast("dict[str, Json]", json.loads(hook.calls.last.request.content))
        (row,) = cast("list[dict[str, Json]]", report["rows"])
        assert report["arr"] == "sonarr"
        assert row["label"] == "Frieren · SubGroup"

    @respx.mock
    def test_a_run_set_not_to_notify_sends_no_wait_webhook(self, tmp_path: Path) -> None:
        hook = respx.post(_WEBHOOK).respond()
        cache = tmp_path / "cache.db"
        _cache_left_with(cache, Arr.SONARR, _pending("h"))
        config = make_config(url="http://sonarr", api_key="key", wait_webhook_url=_WEBHOOK, wait_notify=False)

        _run(
            cache,
            RecordingStrategy(attempts=_ALL_IMPORTED),
            FakeQbit(TORRENT_H_COMPLETE),
            mode=ImportWaitMode.BLOCKING,
            config=config,
        )

        assert hook.call_count == 0
