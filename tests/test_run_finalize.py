# pyright: strict
# pyright: reportPrivateUsage=false
# Drives ArrRunLoop._notify_wait_complete directly (the containment handler has no public
# interface). The repo disables reportPrivateUsage for tests, strict re-flags it.
"""Guards the arr run loop's commit points and its finalize at the arr run's end.

A real run commits after each item, so a stop keeps the items it finished.

The strategy is the shared typed `FakeStrategy` (an `ArrStrategy` with
scripted returns), the arr run loop's collaborators are the small typed
fakes in `run_loop_fakes`, and `_finish_arr_run` is mostly replaced by a typed
recorder - so the contracts are pinned by asserting recorded state.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest

from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.boot_flow import BootFlow
from pearlarr.cache import CacheStore
from pearlarr.config import AppConfig, Arr
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.mappings import TitleMapping
from pearlarr.output import BootStepFinished, Diagnostic, Severity
from pearlarr.progress import ProgressCallback
from pearlarr.skyhook_gateway import SkyHookGateway
from pearlarr.tvdb_orders import AIRED_ORDER, SonarrSeriesOrders
from pearlarr.wait_types import Outcome, OutcomeKind
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import (
    FakeCacheStore,
    ScriptedAniListClient,
    drive_arr_run,
    grab_request,
    make_anilist_gateway,
    make_bare_instance,
    make_config,
    make_run_deps,
    make_services,
)
from .fakes import (
    FakeArrItem,
    FakeClock,
    FakeSkyHookClient,
    FakeStrategy,
    RecordingHub,
    diagnostic_messages,
    install_recording_hub,
)
from .run_loop_fakes import FakeBound, FakeGateway, FinalizeRecorder, make_engine
from .wait_fakes import FakeQbit


class _Staging(FakeStrategy):
    """A library of items 1 and 2, item 1 mapping titles 11 and 12 and item 2 titles 21 and 22.

    Each title's check ends with nothing to grab, which stages the title's cached row and owes the pacing pause, as a
    real check does. The check of title `stop_at` stops the run once its row is staged, as the SIGTERM handler would.
    """

    def __init__(self, services: ArrRunServices, *, stop_at: int | None = None) -> None:
        super().__init__(items=[FakeArrItem(item_id=1), FakeArrItem(item_id=2)], anilist_ids={})
        self._services = services
        self._stop_at = stop_at

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {anilist_id: TitleMapping(anilist_id=anilist_id) for anilist_id in (item.id * 10 + 1, item.id * 10 + 2)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        self._services.report_no_releases(anilist_id, {})
        if anilist_id == self._stop_at:
            raise SystemExit(0)


class _StagingFirstSnapshotFails(_Staging):
    """A `_Staging` library whose item 1 import snapshot fails once its titles are checked."""

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        if item.id == 1:
            raise RuntimeError("snapshot failed")
        return None


class _StopOnSleepClock(FakeClock):
    """A `FakeClock` whose `stop_on`-th sleep raises `SystemExit`, as the SIGTERM handler would mid-pause."""

    def __init__(self, stop_on: int) -> None:
        super().__init__()
        self._stop_on = stop_on

    @override
    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if len(self.sleeps) == self._stop_on:
            raise SystemExit(0)


def _scan_into(
    cache: Path,
    strategy: Callable[[ArrRunServices], _Staging],
    *,
    qbit: FakeQbit | None,
    clock: FakeClock | None = None,
) -> None:
    """One whole Sonarr run of `strategy` over the cache file at `cache`, closed after it as bootstrap closes it.

    `qbit` None is a run with no qBittorrent configured, so a dry run.
    """

    store = CacheStore.load(str(cache), config_hash="")
    try:
        client = None if qbit is None else qbit.client()
        deps = make_run_deps(cache_store=store, qbit=client, clock=clock)
        drive_arr_run(deps, strategy)
    finally:
        store.close()


def _saved_titles(cache: Path) -> list[int]:
    """The titles of the `_Staging` library the cache file at `cache` holds cached rows of, read fresh."""

    store = CacheStore.load(str(cache), config_hash="")
    try:
        return [anilist_id for anilist_id in (11, 12, 21, 22) if store.get_cached_title(Arr.SONARR, anilist_id)]
    finally:
        store.close()


class TestPerItemCommit:
    """A real run commits after each library item, so a stop keeps what finished and rolls back the item in flight.

    A dry run commits nothing.
    """

    def test_a_stop_mid_item_keeps_the_finished_items(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Stopped on item 2's second title, after its first title's row was staged.
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _Staging(services, stop_at=22), qbit=FakeQbit())

        assert _saved_titles(cache) == [11, 12]

    def test_an_item_that_raised_still_commits_what_it_staged(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Item 1's snapshot fails after its titles, an error the scan contains. The scan takes each item's snapshot
        # while imports are waited on, as they are by default. Item 2 stops before reaching its own.
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _StagingFirstSnapshotFails(services, stop_at=21), qbit=FakeQbit())

        assert _saved_titles(cache) == [11, 12]

    def test_a_stop_in_the_pacing_pause_keeps_the_item_before_it(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Every title owes a pause. The second one slept is the one before item 2's first title.
        with pytest.raises(SystemExit):
            _scan_into(cache, _Staging, qbit=FakeQbit(), clock=_StopOnSleepClock(stop_on=2))

        assert _saved_titles(cache) == [11, 12]

    def test_a_dry_run_keeps_nothing(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        _scan_into(cache, _Staging, qbit=None)

        assert _saved_titles(cache) == []


class _GrabRecorder(FakeBound):
    """A fake grab pipeline that counts its `grab_and_cache` calls."""

    def __init__(self) -> None:
        self.calls = 0

    def grab_and_cache(self, req: GrabRequest) -> None:
        del req
        self.calls += 1


class TestGrabPacing:
    """A grab's queries owe the pacing pause, slept before the next title."""

    def test_a_grab_owes_the_pause_the_next_title_sleeps(self) -> None:
        clock = FakeClock()
        pipeline = _GrabRecorder()
        services = make_services(_grab_pipeline=pipeline, _clock=clock, sleep_time=3)

        services.grab_and_cache(grab_request())
        services.pay_pause()
        services.pay_pause()

        assert pipeline.calls == 1
        assert clock.sleeps == [3]


class _RaisingNotifier:
    """A fake `Notifier` whose wait-summary push raises a non-HTTP error."""

    def push_wait_summary(self, *, arr: Arr, result: WaitResult) -> bool:
        del arr, result
        raise RuntimeError("notifier exploded")


class TestNotifyWaitCompleteContainment:
    """A wait-notification failure warns (with traceback) and never propagates."""

    def test_push_failure_is_swallowed_with_a_warning(self) -> None:
        loop = make_bare_instance(
            ArrRunLoop,
            _config=make_config(wait_notify=True),
            _notifier=_RaisingNotifier(),
            _state=ArrRunState(arr=Arr.SONARR),
        )
        recording = install_recording_hub()
        result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

        loop._notify_wait_complete(result)  # must not raise

        (warning,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert warning.message == "Wait completion notification failed unexpectedly - the notification was dropped"
        assert warning.trace is not None


class TestBootNotes:
    """The two prefetch steps' boot step notes must be truthful when a source is unreachable.

    The boot flow emits events, so the finished (label, detail, outcome) is
    read off the recorded `BootStepFinished` stream (conftest's autouse
    teardown uninstalls the hub after every test). SeaDex is the flag-only
    fake. The AniList gateway is real over the shared scripted client, so its
    note reads whether AniList is unreachable the way the run does.
    """

    @staticmethod
    def _step(recording: RecordingHub, label: str) -> BootStepFinished:
        [step] = [e for e in recording.of_type(BootStepFinished) if e.label == label]
        return step

    @staticmethod
    def _run(
        logger: logging.Logger,
        *,
        seadex: FakeGateway | None = None,
        client: ScriptedAniListClient | None = None,
    ) -> RecordingHub:
        strategy = FakeStrategy(
            items=[FakeArrItem()],
            anilist_ids={1: TitleMapping(anilist_id=1)},
        )
        anilist = make_anilist_gateway(client) if client is not None else None
        recording = install_recording_hub()
        make_engine(FinalizeRecorder(), logger, seadex=seadex, anilist=anilist).run_arr(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )
        return recording

    def test_seadex_unreachable_notes_unreachable_not_a_count(self, logger: logging.Logger) -> None:
        # The prefetch "return" is how many ids NEEDED fetching. With SeaDex unreachable
        # none were actually fetched, so a count note would be a lie.
        seadex = FakeGateway()
        seadex.unreachable = True

        step = self._step(self._run(logger, seadex=seadex), "Fetching SeaDex entries")

        assert step.detail == "unreachable"
        assert step.outcome is OutcomeKind.LEFT  # finishes as a warning

    def test_healthy_seadex_prefetch_keeps_the_count_note(self, logger: logging.Logger) -> None:
        step = self._step(self._run(logger), "Fetching SeaDex entries")

        assert step.detail == "cached"  # the fake reports 0 fetched -> cache-warm
        assert step.outcome is OutcomeKind.IMPORTED

    def test_anilist_unreachable_notes_unavailable_not_a_count(self, logger: logging.Logger) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))

        step = self._step(self._run(logger, client=client), "Fetching AniList metadata")

        assert step.detail == "unavailable"
        assert step.outcome is OutcomeKind.LEFT

    def test_healthy_anilist_prefetch_keeps_the_count_note(self, logger: logging.Logger) -> None:
        step = self._step(self._run(logger, client=ScriptedAniListClient()), "Fetching AniList metadata")

        assert step.detail == "1 title"  # the one mapped id needed fetching
        assert step.outcome is OutcomeKind.IMPORTED


# One import event on the scanned item, so the activity read has a checkpoint to commit.
_IMPORT_EVENT = HistoryRecord(
    id=1, date="2026-07-06T10:00:00Z", item_id=3, event_type="downloadFolderImported", download_id=None, reason=None
)


class TestSelectionRecheck:
    """The changed-selection announcement + the full-coverage rule for storing the selection hash.

    Whether the hash was stored is read back through `selection_changed`: after it is stored, any
    OTHER hash reads changed. With no hash stored, nothing reads changed.
    """

    _NOTE = "Selection settings changed - checking cached titles again"

    def _run(
        self,
        logger: logging.Logger,
        *,
        stale: bool = False,
        item_id: int | None = None,
        config: AppConfig | None = None,
        seadex: FakeGateway | None = None,
        anilist: FakeGateway | None = None,
        cap_reached: bool = False,
    ) -> tuple[ArrRunLoop, RecordingHub]:
        recording = install_recording_hub()
        engine = make_engine(FinalizeRecorder(), logger, config=config, seadex=seadex, anilist=anilist)
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=3)],
            anilist_ids={11: TitleMapping(anilist_id=11)},
            services=engine._services if cap_reached else None,
            history=[_IMPORT_EVENT],
        )
        engine._services._selection_changed = stale
        engine.run_arr(strategy, item_id=item_id, dry_run=True, boot=BootFlow())
        return engine, recording

    def _vouched_any(self, engine: ArrRunLoop) -> bool:
        """Whether the arr run stored a selection hash at all (an alien hash then reads changed)."""

        return engine.cache_store.selection_changed(Arr.SONARR, "some-other-digest")

    def test_selection_changed_is_announced(self, logger: logging.Logger) -> None:
        _, recording = self._run(logger, stale=True)

        assert self._NOTE in diagnostic_messages(recording)

    def test_unchanged_selection_stays_quiet(self, logger: logging.Logger) -> None:
        _, recording = self._run(logger, stale=False)

        assert self._NOTE not in diagnostic_messages(recording)

    def test_single_item_run_does_not_announce(self, logger: logging.Logger) -> None:
        # A single-item run re-checks only its id and cannot store the selection hash, so the whole-
        # library note is suppressed (it would overstate the scan and recur).
        _, recording = self._run(logger, stale=True, item_id=3)

        assert self._NOTE not in diagnostic_messages(recording)

    def test_ignore_update_times_suppresses_the_announcement(self, logger: logging.Logger) -> None:
        # The ignore flag already re-checks everything. Announcing a selection
        # re-check on top would be noise.
        _, recording = self._run(logger, stale=True, config=make_config(ignore_seadex_update_times=True))

        assert self._NOTE not in diagnostic_messages(recording)

    def test_full_run_stores_the_current_selection_hash(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger)

        assert engine.cache_store.selection_changed(Arr.SONARR, engine._config.selection_hash()) is False
        assert self._vouched_any(engine) is True

    def test_single_item_run_does_not_store_the_selection_hash(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger, item_id=3)

        assert self._vouched_any(engine) is False

    def test_a_run_that_held_a_title_past_the_cap_stores_the_selection_hash_and_commits_the_checkpoint(
        self, logger: logging.Logger
    ) -> None:
        # A hold keeps neither the selection hash nor the checkpoint back.
        engine, _ = self._run(logger, cap_reached=True)

        assert self._vouched_any(engine) is True
        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is not None

    def test_unreachable_run_does_not_store_the_selection_hash(self, logger: logging.Logger) -> None:
        unreachable_gateway = FakeGateway()
        unreachable_gateway.unreachable = True
        engine, _ = self._run(logger, seadex=unreachable_gateway)

        assert self._vouched_any(engine) is False

    def test_an_anilist_unreachable_run_stores_the_selection_hash_and_commits_the_checkpoint(
        self, logger: logging.Logger
    ) -> None:
        # A title AniList left unread isn't cached, so AniList being unreachable keeps neither back.
        unreachable_gateway = FakeGateway()
        unreachable_gateway.unreachable = True
        engine, _ = self._run(logger, anilist=unreachable_gateway)

        assert self._vouched_any(engine) is True
        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is not None


class _OrderFetchingStrategy(FakeStrategy):
    """Warms episodes as the Sonarr strategy does, its order fetch asking `skyhook` for item 3 on a v4 Sonarr."""

    def __init__(self, skyhook: SkyHookGateway) -> None:
        super().__init__(
            items=[FakeArrItem(item_id=3)],
            anilist_ids={11: TitleMapping(anilist_id=11)},
            history=[_IMPORT_EVENT],
        )
        self._skyhook = skyhook

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_orders(self, items: list[FakeArrItem], *, progress: ProgressCallback | None = None) -> int:
        series = [SonarrSeriesOrders(item.id, AIRED_ORDER, None) for item in items]
        return self._skyhook.prefetch(series, dry_run=True, progress=progress)


@pytest.mark.parametrize(
    ("failing", "note", "full_scan"),
    [
        pytest.param(frozenset({(3, "official")}), "unreachable", False, id="outage"),
        pytest.param(frozenset[tuple[int, str]](), "1 series", True, id="healthy"),
    ],
)
def test_a_skyhook_unreachable_notes_unreachable_and_the_run_neither_stores_the_selection_hash_nor_commits(
    logger: logging.Logger, failing: frozenset[tuple[int, str]], note: str, full_scan: bool
) -> None:
    # A series SkyHook left unrefreshed was read by its old stored orders, so the run commits neither and the next one
    # repeats the re-checks a settings or file change drove.
    recording = install_recording_hub()
    store = FakeCacheStore()
    skyhook = SkyHookGateway(client=FakeSkyHookClient(failing=failing), cache_store=store)
    engine = make_engine(FinalizeRecorder(), logger, cache_store=store, skyhook=skyhook)

    engine.run_arr(_OrderFetchingStrategy(skyhook), item_id=None, dry_run=True, boot=BootFlow())

    [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching TVDB orders"]
    assert step.detail == note
    assert store.selection_changed(Arr.SONARR, "some-other-digest") is full_scan
    assert (store.get_history_checkpoint(Arr.SONARR) is not None) is full_scan
