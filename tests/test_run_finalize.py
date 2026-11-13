# pyright: strict
# pyright: reportPrivateUsage=false
# Drives LegLoop._notify_wait_complete directly (the containment arm has no public
# seam). The repo disables reportPrivateUsage for tests, strict re-flags it.
"""Guards the run loop's commit points and its single end-of-run finalize site.

A title held past `max_torrents_to_add` never stops the scan, and the ONE
post-loop `_finalize_run` site runs once on every path. These pin that hoist so
a future change can't silently double-finalize or skip the blocking/import pass.
A real run also commits after each title, so a stop keeps the titles it finished.

The strategy is the shared typed `FakeStrategy` (an `ArrSync` recording
its `process_al_id` calls), the engine's collaborators are small typed fakes,
and `_finalize_run` is mostly replaced by a typed recorder - so the contracts are
pinned by asserting recorded state.
"""

import logging
from pathlib import Path
from typing import override

import pytest

from pearlarr.anilist_gateway import AniListGateway
from pearlarr.arr_models import HistoryRecord
from pearlarr.boot_flow import BootFlow
from pearlarr.cache import AbstractCacheStore, CacheStore
from pearlarr.config import AppConfig, Arr
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.import_wait import ImportWaitManager
from pearlarr.leg_state import RunContext
from pearlarr.mappings import TitleMapping
from pearlarr.output import BootStepFinished, CountsMark, Diagnostic, EntryHeader, Severity, SeverityCounts
from pearlarr.output.events import EntryState
from pearlarr.output.recording import RecordingHub
from pearlarr.progress import ProgressSink
from pearlarr.protocols import ImportCompleter
from pearlarr.run_loop import LegLoop
from pearlarr.run_services import LegServices
from pearlarr.skyhook_gateway import SkyHookGateway
from pearlarr.tvdb_orders import OFFICIAL_ORDER, SeriesOrderFacts
from pearlarr.wait_types import ImportWaitMode, Outcome, OutcomeKind
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import (
    CLIENT_SENTINEL,
    FakeCacheStore,
    ScriptedAniListClient,
    grab_request,
    make_anilist_gateway,
    make_bare_instance,
    make_config,
    make_import_wait_manager,
    make_services,
    pending_import,
)
from .fakes import FakeArrItem, FakeClock, FakeSkyHookClient, FakeStrategy, install_recording_hub


class _FakeGateway:
    """Stands in for the AniList/SeaDex gateways: a cache-warm no-op (0 fetched)."""

    # The loop reads each gateway's outage flag. Never down unless a test sets it.
    outage: bool = False

    def load_cache(self) -> None:
        pass

    def prefetch(
        self,
        ids: object,
        *,
        preview: bool = False,
        progress: ProgressSink | None = None,
    ) -> int:
        del ids, preview, progress
        return 0


class _FakeReporter:
    """The run-loop log hooks the scan and the run tail drive (each just acknowledges)."""

    def counts_mark(self) -> CountsMark:
        return SeverityCounts().bound_mark()

    def log_arr_start(self, arr: Arr, n_items: int) -> bool:
        return True

    def log_arr_item_start(self, arr: Arr, item_title: str, n_item: int, n_items: int) -> bool:
        return True

    def log_arr_item_unmonitored(self, ctx: RunContext, item_title: str) -> bool:
        return True

    def log_no_anilist_mappings(self, ctx: RunContext, title: str) -> bool:
        return True

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        return True

    def scan_finished(self, arr: Arr) -> bool:
        return True

    def log_run_summary(self, ctx: RunContext, *, preview: bool, has_client: bool) -> bool:
        return True

    def run_finished(self, arr: Arr) -> bool:
        return True


class _FakeBound:
    """A collaborator whose only run-loop hook is the `begin_run` ctx bind."""

    def bind_leg(self, state: RunContext, strategy: ImportCompleter | None = None) -> None:
        del state, strategy


class _FinalizeRecorder:
    """A typed stand-in for `RunLoop._finalize_run` that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _services(*, qbit: object = None, **overrides: object) -> LegServices:
    """A bare real `RunServices` whose per-id collaborators are ctx-bind fakes. `qbit` set makes the run real."""

    return make_services(
        qbit=qbit,
        _filter=_FakeBound(),
        _grab_pipeline=_FakeBound(),
        _records=_FakeBound(),
        _reporter=_FakeReporter(),
        **overrides,
    )


def _engine(
    finalize: _FinalizeRecorder | None,
    logger: logging.Logger,
    *,
    config: AppConfig | None = None,
    seadex: _FakeGateway | None = None,
    cache_store: AbstractCacheStore | None = None,
    anilist: AniListGateway | _FakeGateway | None = None,
    services: LegServices | None = None,
    wait_manager: _FakeBound | ImportWaitManager | None = None,
    skyhook: SkyHookGateway | None = None,
) -> LegLoop:
    """A bare `RunLoop` wired with typed fakes for the run-loop collaborators.

    The strategy reaches `run_sync` typed (it's an `ArrSync`). The rest are
    injected as bare attributes (the methods only read them), and `_finalize_run`
    is shadowed by the recorder so the single finalize site is observable
    (`None` keeps the real one). The `_services` hub defaults to a preview
    `_services()` (the loop reads its `arr`, `begin_run`, `mark_dirty`,
    `is_preview` and `pay_pause`). The `cache_store` backs the activity scan's checkpoint.
    `config` overrides the loop's config (the activity-scan toggle tests).
    `cache_store` shares one store across engines (the checkpoint replay tests).
    `anilist` swaps in a real gateway (the AniList boot-note tests) or a fake that is down (the vouch test).
    `services` swaps the hub (the commit tests make it real and share its store).
    `wait_manager` swaps the ctx-bind wait manager (the contained-error commit test, the run-start test).
    `skyhook` swaps in a gateway over a failing client (the SkyHook outage tests).
    """

    config = config if config is not None else make_config()
    store = cache_store if cache_store is not None else FakeCacheStore()
    services = services if services is not None else _services()
    shadowed = {} if finalize is None else {"_finalize_run": finalize}
    return make_bare_instance(
        LegLoop,
        qbit=services.qbit,
        logger=logger,
        _config=config,
        _arr_config=config.for_arr(Arr.SONARR),
        _anilist=anilist if anilist is not None else _FakeGateway(),
        _seadex=seadex if seadex is not None else _FakeGateway(),
        _skyhook=skyhook if skyhook is not None else SkyHookGateway(client=FakeSkyHookClient(), cache_store=store),
        _reporter=_FakeReporter(),
        _services=services,
        _wait_manager=wait_manager if wait_manager is not None else _FakeBound(),
        cache_store=store,
        **shadowed,
    )


class TestHeldTitleKeepsScanning:
    """A title held past the run cap stops nothing: every item is scanned and the single finalize site runs once."""

    def test_held_title_keeps_scanning_and_finalizes_once(self, logger: logging.Logger) -> None:
        finalize = _FinalizeRecorder()
        engine = _engine(finalize, logger)
        # Every id is held by the cap, as the pipeline holds a grabbable release past it.
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={1: TitleMapping(anilist_id=1)},
            held_through=engine._services,
        )

        engine.run_leg(strategy, item_id=None, dry_run=True, boot=BootFlow())

        # The hold on the first item never stopped the scan: the second item's id is still checked.
        assert strategy.process_calls == [1, 1]
        assert engine._services.ctx.stats.held_by_cap == 2
        # ...and the single post-loop finalize ran exactly once.
        assert finalize.calls == 1


class TestPerIdErrorContainment:
    """One AniList id's failure is contained to that id, not the whole item."""

    def test_one_al_id_error_does_not_skip_siblings(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: TitleMapping(anilist_id=1), 2: TitleMapping(anilist_id=2)},
            process_raises_on=1,
        )
        finalize = _FinalizeRecorder()
        recording = install_recording_hub()
        _engine(finalize, logger).run_leg(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # The first id raised but was contained: the sibling id 2 is still processed.
        assert strategy.process_calls == [1, 2]
        # The per-id failure is an ERROR Diagnostic with its traceback (containment
        # is observable).
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == "A (AniList #1): unexpected error (boom on al_id 1) - skipping this AniList id"
        assert error.trace is not None
        # The single finalize still ran once on the normal end-of-run path.
        assert finalize.calls == 1


class _ItemRaisingStrategy(FakeStrategy):
    """A `FakeStrategy` whose mapping lookup raises for one item id.

    This is the seam `run_sync` hits per item BEFORE the per-id loop (the
    OUTER containment arm).
    """

    def __init__(
        self,
        *,
        items: list[FakeArrItem],
        anilist_ids: dict[int, TitleMapping],
        raise_on_item: int,
    ) -> None:
        super().__init__(items=items, anilist_ids=anilist_ids)
        self._raise_on_item = raise_on_item

    @override
    def item_anilist_ids(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        # Only the per-item scan calls with the log_ignored default. The pre-loop
        # activity/prefetch passes pass False and must stay healthy.
        if log_ignored and item.id == self._raise_on_item:
            raise RuntimeError("mapping lookup exploded")
        return super().item_anilist_ids(item, log_ignored)


class TestPerItemErrorContainment:
    """A whole item's failure is contained to that item, not the run."""

    def test_one_item_error_does_not_skip_siblings(self, logger: logging.Logger) -> None:
        strategy = _ItemRaisingStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={5: TitleMapping(anilist_id=5)},
            raise_on_item=1,
        )
        finalize = _FinalizeRecorder()
        recording = install_recording_hub()

        _engine(finalize, logger).run_leg(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # Item A raised before its per-id loop began. Sibling item B still processed.
        assert strategy.process_calls == [5]
        # The per-item failure is an ERROR Diagnostic with its traceback.
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == "A: unexpected error (mapping lookup exploded) - skipping this title"
        assert error.trace is not None
        # The run still reached the single post-loop finalize site.
        assert finalize.calls == 1


class _StagingStrategy(FakeStrategy):
    """A `FakeStrategy` whose items each map ids 1 and 2, every id staging a row through `no_releases_skip`.

    A row is keyed `item.id * 10 + al_id`. `stop_at` (item id, AniList id) raises `SystemExit` after that id's
    row, standing in for the SIGTERM handler. `None` never stops.
    """

    def __init__(self, *, items: list[FakeArrItem], services: LegServices, stop_at: tuple[int, int] | None) -> None:
        super().__init__(items=items, anilist_ids={1: TitleMapping(anilist_id=1), 2: TitleMapping(anilist_id=2)})
        self._services = services
        self._stop_at = stop_at

    @override
    def check_title(self, item: FakeArrItem, al_id: int, mapping: TitleMapping) -> None:
        self._services.no_releases_skip(item.id * 10 + al_id, {"name": item.title})
        if (item.id, al_id) == self._stop_at:
            raise SystemExit(0)

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        return item.id


class _RaisingSnapshots(_FakeBound):
    """A wait manager whose per-item pending snapshot raises, after that item's ids staged their rows."""

    def snapshot_pending_for_series(self, series_id: int) -> None:
        raise RuntimeError(f"snapshot failed for series {series_id}")


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


_TITLES = [FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")]


def _kept(path: Path, keys: list[int]) -> list[int]:
    """The `keys` whose entry rows the cache at `path` holds once reopened."""

    reopened = CacheStore.load(str(path), config_hash="")
    try:
        return [key for key in keys if reopened.get_entry(Arr.SONARR, key) is not None]
    finally:
        reopened.close()


def _run(engine: LegLoop, strategy: _StagingStrategy, mode: ImportWaitMode = ImportWaitMode.OFF) -> None:
    """One full-library `run_sync` of `strategy` with wait mode `mode`."""

    engine.run_leg(strategy, item_id=None, dry_run=False, import_wait_mode=mode, boot=BootFlow())


class TestPerTitleCommit:
    """A real run commits between titles, so a stop keeps what finished and rolls back the title in flight.

    A preview run commits nothing.
    """

    def test_a_stop_mid_title_keeps_the_finished_titles(self, logger: logging.Logger, tmp_path: Path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.db"), config_hash="")
        services = _services(qbit=CLIENT_SENTINEL, cache_store=store)
        engine = _engine(_FinalizeRecorder(), logger, cache_store=store, services=services)

        # Stopped on title B's second id, after its first id's row was staged.
        with pytest.raises(SystemExit):
            _run(engine, _StagingStrategy(items=_TITLES, services=services, stop_at=(2, 2)))
        store.close()

        assert _kept(tmp_path / "cache.db", [11, 12, 21, 22]) == [11, 12]

    def test_a_title_that_raised_still_commits_what_it_staged(self, logger: logging.Logger, tmp_path: Path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.db"), config_hash="")
        services = _services(qbit=CLIENT_SENTINEL, cache_store=store)
        engine = _engine(
            _FinalizeRecorder(), logger, cache_store=store, services=services, wait_manager=_RaisingSnapshots()
        )

        # Title A's snapshot raises after its ids, a contained title-level error. B stops before reaching its own.
        with pytest.raises(SystemExit):
            _run(engine, _StagingStrategy(items=_TITLES, services=services, stop_at=(2, 1)), ImportWaitMode.BLOCKING)
        store.close()

        assert _kept(tmp_path / "cache.db", [11, 12, 21]) == [11, 12]

    def test_a_stop_in_the_pacing_pause_keeps_the_title_before_it(self, logger: logging.Logger, tmp_path: Path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.db"), config_hash="")
        # Every id owes a pause. The second one slept is the one after title A's last id.
        clock = _StopOnSleepClock(stop_on=2)
        services = _services(qbit=CLIENT_SENTINEL, cache_store=store, _clock=clock, sleep_time=1)
        engine = _engine(_FinalizeRecorder(), logger, cache_store=store, services=services)

        with pytest.raises(SystemExit):
            _run(engine, _StagingStrategy(items=_TITLES, services=services, stop_at=None))
        store.close()

        assert clock.sleeps == [1, 1]
        assert _kept(tmp_path / "cache.db", [11, 12, 21]) == [11, 12]

    def test_a_preview_run_keeps_nothing(self, logger: logging.Logger, tmp_path: Path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.db"), config_hash="")
        services = _services(cache_store=store)
        engine = _engine(_FinalizeRecorder(), logger, cache_store=store, services=services)

        _run(engine, _StagingStrategy(items=_TITLES[:1], services=services, stop_at=None))
        store.close()

        assert _kept(tmp_path / "cache.db", [11, 12]) == []

    @pytest.mark.parametrize(("qbit", "kept"), [(CLIENT_SENTINEL, [1]), (None, [])], ids=["real", "preview"])
    def test_the_finalize_save_commits_only_a_real_run(
        self, logger: logging.Logger, tmp_path: Path, qbit: object, kept: list[int]
    ) -> None:
        store = CacheStore.load(str(tmp_path / "cache.db"), config_hash="")
        services = _services(qbit=qbit, cache_store=store)
        engine = _engine(None, logger, cache_store=store, services=services)
        # With no titles, the run tail's save is the run's only commit.
        store.update_cache(Arr.SONARR, 1, {"name": "A"})

        _run(engine, _StagingStrategy(items=[], services=services, stop_at=None))
        store.close()

        assert _kept(tmp_path / "cache.db", [1]) == kept


class TestRunStartKeepsPendingRecords:
    """Nothing at a run's start drops a carried-over record, however old."""

    def test_a_years_old_record_survives_the_run_start_and_is_reported(self, logger: logging.Logger) -> None:
        store = FakeCacheStore()
        old = pending_import(infohash="h", series_id=1, added_at="2000-01-01 00:00:00")
        store.put_pending(Arr.SONARR, old.infohash, old.to_json())
        services = _services(qbit=CLIENT_SENTINEL, cache_store=store)
        manager = make_import_wait_manager(cache_store=store)
        engine = _engine(_FinalizeRecorder(), logger, cache_store=store, services=services, wait_manager=manager)
        recording = install_recording_hub()

        _run(engine, _StagingStrategy(items=_TITLES[:1], services=services, stop_at=None), ImportWaitMode.BLOCKING)

        assert set(store.get_pending(Arr.SONARR)) == {"h"}
        assert [(h.state, h.title) for h in recording.of_type(EntryHeader)] == [(EntryState.QUEUED, old.display_label)]


class _GrabRecorder(_FakeBound):
    """A grab pipeline stand-in that counts its `grab_and_cache` calls."""

    def __init__(self) -> None:
        self.calls = 0

    def grab_and_cache(self, req: GrabRequest) -> None:
        del req
        self.calls += 1


class TestGrabPacing:
    """A grab's queries owe the pacing pause, slept before the next id."""

    def test_a_grab_owes_the_pause_the_next_id_sleeps(self) -> None:
        clock = FakeClock()
        pipeline = _GrabRecorder()
        services = make_services(_grab_pipeline=pipeline, _clock=clock, sleep_time=3)

        services.grab_and_cache(grab_request())
        services.pay_pause()
        services.pay_pause()

        assert pipeline.calls == 1
        assert clock.sleeps == [3]


class _RaisingNotifier:
    """A `Notifier` stand-in whose wait-summary push raises a non-HTTP error."""

    def push_wait_summary(self, *, arr: Arr, result: WaitResult) -> bool:
        del arr, result
        raise RuntimeError("notifier exploded")


class TestNotifyWaitCompleteContainment:
    """A wait-notification failure warns (with traceback) and never propagates."""

    def test_push_failure_is_swallowed_with_a_warning(self) -> None:
        loop = make_bare_instance(
            LegLoop,
            _config=make_config(wait_notify=True),
            _notifier=_RaisingNotifier(),
            _ctx=RunContext(arr=Arr.SONARR),
        )
        recording = install_recording_hub()
        result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

        loop._notify_wait_complete(result)  # must not raise

        (warning,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert warning.message == "Wait completion notification failed unexpectedly - the notification was dropped"
        assert warning.trace is not None


class TestBootNotes:
    """The two prefetch steps' ledger notes must be truthful on an outage.

    The boot flow emits events, so the graduated (label, detail, outcome) is
    read off the recorded `BootStepFinished` stream (conftest's autouse
    teardown uninstalls the hub after every test). SeaDex is the flag-only
    fake. The AniList gateway is real over the shared scripted client, so its
    note reads the breaker the way the run does.
    """

    @staticmethod
    def _step(recording: RecordingHub, label: str) -> BootStepFinished:
        [step] = [e for e in recording.of_type(BootStepFinished) if e.label == label]
        return step

    @staticmethod
    def _run(
        logger: logging.Logger,
        *,
        seadex: _FakeGateway | None = None,
        client: ScriptedAniListClient | None = None,
    ) -> RecordingHub:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: TitleMapping(anilist_id=1)},
        )
        anilist = make_anilist_gateway(client) if client is not None else None
        recording = install_recording_hub()
        _engine(_FinalizeRecorder(), logger, seadex=seadex, anilist=anilist).run_leg(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )
        return recording

    def test_seadex_outage_notes_unreachable_not_a_count(self, logger: logging.Logger) -> None:
        # The prefetch "return" is how many ids NEEDED fetching. On an outage
        # none were actually fetched, so a count note would be a lie.
        seadex = _FakeGateway()
        seadex.outage = True

        step = self._step(self._run(logger, seadex=seadex), "Fetching SeaDex entries")

        assert step.detail == "unreachable"
        assert step.outcome is OutcomeKind.LEFT  # graduates as a warning

    def test_healthy_seadex_prefetch_keeps_the_count_note(self, logger: logging.Logger) -> None:
        step = self._step(self._run(logger), "Fetching SeaDex entries")

        assert step.detail == "cached"  # the fake reports 0 fetched -> cache-warm
        assert step.outcome is OutcomeKind.IMPORTED

    def test_anilist_outage_notes_unavailable_not_a_count(self, logger: logging.Logger) -> None:
        client = ScriptedAniListClient(failing=frozenset({1}))

        step = self._step(self._run(logger, client=client), "Fetching AniList metadata")

        assert step.detail == "unavailable"
        assert step.outcome is OutcomeKind.LEFT

    def test_healthy_anilist_prefetch_keeps_the_count_note(self, logger: logging.Logger) -> None:
        step = self._step(self._run(logger, client=ScriptedAniListClient()), "Fetching AniList metadata")

        assert step.detail == "1 entry"  # the one mapped id needed fetching
        assert step.outcome is OutcomeKind.IMPORTED


class TestScanItemContext:
    """The scan seeds the run context with the arr item's own title before any per-id line."""

    def test_arr_title_follows_the_item(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: TitleMapping(anilist_id=1)},
        )
        engine = _engine(_FinalizeRecorder(), logger)
        install_recording_hub()

        engine.run_leg(strategy, item_id=None, dry_run=True, boot=BootFlow())

        assert engine._ctx.arr_title == "A"


# One import event on the scanned item, so the activity scan has a checkpoint to commit.
_IMPORT_EVENT = HistoryRecord(
    id=1, date="2026-07-06T10:00:00Z", item_id=3, event_type="downloadFolderImported", download_id=None, reason=None
)


class TestSelectionRecheck:
    """The stale-selection announcement + the full-coverage vouch rule.

    Vouch state is read back through `selection_stale`: after a vouch, any
    OTHER digest reads stale. With nothing vouched, everything reads fresh.
    """

    _NOTE = "Matching settings changed - rechecking cached entries"

    def _run(
        self,
        logger: logging.Logger,
        *,
        stale: bool = False,
        item_id: int | None = None,
        config: AppConfig | None = None,
        seadex: _FakeGateway | None = None,
        anilist: _FakeGateway | None = None,
        held: bool = False,
    ) -> tuple[LegLoop, RecordingHub]:
        recording = install_recording_hub()
        engine = _engine(_FinalizeRecorder(), logger, config=config, seadex=seadex, anilist=anilist)
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=3, title="A")],
            anilist_ids={11: TitleMapping(anilist_id=11)},
            held_through=engine._services if held else None,
            history=[_IMPORT_EVENT],
        )
        engine._services._selection_stale = stale
        engine.run_leg(strategy, item_id=item_id, dry_run=True, boot=BootFlow())
        return engine, recording

    def _notes(self, recording: RecordingHub) -> list[str]:
        return [d.message for d in recording.of_type(Diagnostic)]

    def _vouched_any(self, engine: LegLoop) -> bool:
        """Whether the run recorded a digest at all (an alien digest then reads stale)."""

        return engine.cache_store.selection_changed(Arr.SONARR, "some-other-digest")

    def test_stale_selection_is_announced(self, logger: logging.Logger) -> None:
        _, recording = self._run(logger, stale=True)

        assert self._NOTE in self._notes(recording)

    def test_fresh_selection_stays_quiet(self, logger: logging.Logger) -> None:
        _, recording = self._run(logger, stale=False)

        assert self._NOTE not in self._notes(recording)

    def test_single_item_run_does_not_announce(self, logger: logging.Logger) -> None:
        # A single-item run re-checks only its id and cannot vouch, so the whole-
        # library note is suppressed (it would overstate the pass and recur).
        _, recording = self._run(logger, stale=True, item_id=3)

        assert self._NOTE not in self._notes(recording)

    def test_ignore_update_times_suppresses_the_announcement(self, logger: logging.Logger) -> None:
        # The ignore flag already re-checks everything. Announcing a selection
        # re-check on top would be noise.
        _, recording = self._run(logger, stale=True, config=make_config(ignore_seadex_update_times=True))

        assert self._NOTE not in self._notes(recording)

    def test_full_run_vouches_the_current_digest(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger)

        assert engine.cache_store.selection_changed(Arr.SONARR, engine._config.selection_hash()) is False
        assert self._vouched_any(engine) is True

    def test_single_item_run_does_not_vouch(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger, item_id=3)

        assert self._vouched_any(engine) is False

    def test_a_run_that_held_a_title_past_the_cap_vouches_and_commits_the_checkpoint(
        self, logger: logging.Logger
    ) -> None:
        # A hold keeps neither the vouch nor the checkpoint back.
        engine, _ = self._run(logger, held=True)

        assert self._vouched_any(engine) is True
        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is not None

    def test_outage_run_does_not_vouch(self, logger: logging.Logger) -> None:
        outage = _FakeGateway()
        outage.outage = True
        engine, _ = self._run(logger, seadex=outage)

        assert self._vouched_any(engine) is False

    def test_an_anilist_outage_run_vouches_and_commits_the_checkpoint(self, logger: logging.Logger) -> None:
        # A title AniList left unread isn't cached as done, so the outage keeps neither back.
        outage = _FakeGateway()
        outage.outage = True
        engine, _ = self._run(logger, anilist=outage)

        assert self._vouched_any(engine) is True
        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is not None


class _OrderFetchingStrategy(FakeStrategy):
    """Warms episodes as the Sonarr strategy does, its order fetch asking `skyhook` for item 3 on a v4 Sonarr."""

    def __init__(self, skyhook: SkyHookGateway) -> None:
        super().__init__(
            items=[FakeArrItem(item_id=3, title="A")],
            anilist_ids={11: TitleMapping(anilist_id=11)},
            history=[_IMPORT_EVENT],
        )
        self._skyhook = skyhook

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_orders(self, items: list[FakeArrItem], *, progress: ProgressSink | None = None) -> int:
        series = [SeriesOrderFacts(item.id, OFFICIAL_ORDER, None) for item in items]
        return self._skyhook.prefetch(series, preview=True, progress=progress)


@pytest.mark.parametrize(
    ("failing", "note", "full_scan"),
    [
        pytest.param(frozenset({(3, "official")}), "unreachable", False, id="outage"),
        pytest.param(frozenset[tuple[int, str]](), "1 series", True, id="healthy"),
    ],
)
def test_a_skyhook_outage_notes_unreachable_and_the_run_neither_vouches_nor_commits(
    logger: logging.Logger, failing: frozenset[tuple[int, str]], note: str, full_scan: bool
) -> None:
    # A series the outage left unrefreshed was read by its old order record, so the pass stays open and the next one
    # repeats the re-checks a settings or file change drove.
    recording = install_recording_hub()
    store = FakeCacheStore()
    skyhook = SkyHookGateway(client=FakeSkyHookClient(failing=failing), cache_store=store)
    engine = _engine(_FinalizeRecorder(), logger, cache_store=store, skyhook=skyhook)

    engine.run_leg(_OrderFetchingStrategy(skyhook), item_id=None, dry_run=True, boot=BootFlow())

    [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching TVDB orders"]
    assert step.detail == note
    assert store.selection_changed(Arr.SONARR, "some-other-digest") is full_scan
    assert (store.get_history_checkpoint(Arr.SONARR) is not None) is full_scan
