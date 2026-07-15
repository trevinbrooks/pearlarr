# pyright: strict
# pyright: reportPrivateUsage=false
# Drives RunLoop._notify_wait_complete directly (the containment arm has no public
# seam); the repo disables reportPrivateUsage for tests, strict re-flags it.
"""Guards the single end-of-run finalize site.

When `max_torrents_to_add` is reached mid-run, `_grab` returns a pure bool
(it no longer finalizes); `run_sync` breaks the per-item scan and runs the ONE
post-loop `_finalize_run` site - the same site the normal end-of-run path
reaches. These pin both halves of that hoist so a future change can't silently
double-finalize or skip the blocking/import pass on the cap-reached break.

The strategy is the shared typed `FakeStrategy` (an `ArrSync` recording
its `process_al_id` calls), the engine's collaborators are small typed fakes,
and `_finalize_run` is replaced by a typed recorder - so the contracts are
pinned by asserting recorded state.
"""

import logging
from typing import override

from pearlarr.boot_flow import BootFlow
from pearlarr.config import AppConfig, Arr
from pearlarr.manual_import import Outcome, OutcomeCategory
from pearlarr.mappings import MappingEntry
from pearlarr.output import BootStepFinished, CountsMark, Diagnostic, Severity, SeverityCounts
from pearlarr.output.recording import RecordingHub
from pearlarr.protocols import ImportCompleter
from pearlarr.reporter import RunContext
from pearlarr.run_loop import RunLoop
from pearlarr.seadex_types import ProgressSink
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import FakeCacheStore, make_bare_instance, make_config, make_services
from .fakes import FakeArrItem, FakeStrategy, install_recording_hub


class _FakeGateway:
    """Stands in for the AniList/SeaDex gateways: a cache-warm no-op (0 fetched)."""

    # The loop reads the SeaDex outage flag for the boot note; never down here.
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
    """The run-loop log hooks the scan drives (each just acknowledges)."""

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


class _FakeBound:
    """A collaborator whose only run-loop hook is the `begin_run` ctx bind."""

    def begin_run(self, ctx: RunContext, strategy: ImportCompleter | None = None) -> None:
        del ctx, strategy


class _FinalizeRecorder:
    """A typed stand-in for `RunLoop._finalize_run` that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _engine(
    finalize: _FinalizeRecorder,
    logger: logging.Logger,
    *,
    config: AppConfig | None = None,
    seadex: _FakeGateway | None = None,
) -> RunLoop:
    """A bare `RunLoop` wired with typed fakes for the run-loop collaborators.

    The strategy reaches `run_sync` typed (it's an `ArrSync`); the rest are
    injected as bare attributes (the methods only read them), and `_finalize_run`
    is shadowed by the recorder so the single finalize site is observable. The
    `_services` hub is a bare real `RunServices` (the loop reads its `arr`,
    `begin_run`, `mark_dirty` and `is_preview`) whose per-id collaborators are
    ctx-bind fakes; the `cache_store` backs the activity scan's checkpoint.
    `config` overrides the loop's config (the activity-scan toggle tests).
    """

    config = config if config is not None else make_config()
    services = make_services(
        qbit=None,
        _filter=_FakeBound(),
        _grab_pipeline=_FakeBound(),
    )
    return make_bare_instance(
        RunLoop,
        qbit=None,
        logger=logger,
        _config=config,
        _arr_config=config.for_arr(Arr.SONARR),
        _anilist=_FakeGateway(),
        _seadex=seadex if seadex is not None else _FakeGateway(),
        _reporter=_FakeReporter(),
        _services=services,
        _wait_manager=_FakeBound(),
        _finalize_run=finalize,
        cache_store=FakeCacheStore(),
    )


class TestCapReachedFinalizesOnce:
    """A mid-run cap stops the scan and finalizes exactly once, at the single site."""

    def test_cap_reached_breaks_loop_and_finalizes_once(self, logger: logging.Logger) -> None:
        # Cap reached on the first id: process_al_id returns True (stop the run).
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={1: MappingEntry(anilist_id=1)},
            process_returns=True,
        )
        finalize = _FinalizeRecorder()

        _engine(finalize, logger).run_sync(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # The cap stopped the scan after the first id: the second item is never reached.
        assert strategy.process_calls == [1]
        # ...and the single post-loop finalize ran exactly once.
        assert finalize.calls == 1


class TestPerIdErrorContainment:
    """One AniList id's failure is contained to that id, not the whole item."""

    def test_one_al_id_error_does_not_skip_siblings(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: MappingEntry(anilist_id=1), 2: MappingEntry(anilist_id=2)},
            process_raises_on=1,
        )
        finalize = _FinalizeRecorder()
        recording = install_recording_hub()
        _engine(finalize, logger).run_sync(
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
        anilist_ids: dict[int, MappingEntry],
        raise_on_item: int,
    ) -> None:
        super().__init__(items=items, anilist_ids=anilist_ids)
        self._raise_on_item = raise_on_item

    @override
    def item_anilist_ids(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, MappingEntry]:
        # Only the per-item scan calls with the log_ignored default; the pre-loop
        # activity/prefetch passes pass False and must stay healthy.
        if log_ignored and item.id == self._raise_on_item:
            raise RuntimeError("mapping lookup exploded")
        return super().item_anilist_ids(item, log_ignored)


class TestPerItemErrorContainment:
    """A whole item's failure is contained to that item, not the run."""

    def test_one_item_error_does_not_skip_siblings(self, logger: logging.Logger) -> None:
        strategy = _ItemRaisingStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={5: MappingEntry(anilist_id=5)},
            raise_on_item=1,
        )
        finalize = _FinalizeRecorder()
        recording = install_recording_hub()

        _engine(finalize, logger).run_sync(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # Item A raised before its per-id loop began; sibling item B still processed.
        assert strategy.process_calls == [5]
        # The per-item failure is an ERROR Diagnostic with its traceback.
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == "A: unexpected error (mapping lookup exploded) - skipping this title"
        assert error.trace is not None
        # The run still reached the single post-loop finalize site.
        assert finalize.calls == 1


class _RaisingNotifier:
    """A `Notifier` stand-in whose wait-summary push raises a non-HTTP error."""

    def push_wait_summary(self, *, arr: Arr, result: WaitResult) -> bool:
        del arr, result
        raise RuntimeError("notifier exploded")


class TestNotifyWaitCompleteContainment:
    """A wait-notification failure warns (with traceback) and never propagates."""

    def test_push_failure_is_swallowed_with_a_warning(self) -> None:
        loop = make_bare_instance(
            RunLoop,
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


class TestSeaDexBootNote:
    """The SeaDex prefetch step's ledger note must be truthful on an outage.

    The boot flow emits events, so the graduated (label, detail, outcome) is
    read off the recorded `BootStepFinished` stream (conftest's autouse
    teardown uninstalls the hub after every test).
    """

    def _seadex_step(self, recording: RecordingHub) -> BootStepFinished:
        [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching SeaDex entries"]
        return step

    def _run(self, logger: logging.Logger, *, seadex: _FakeGateway) -> RecordingHub:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: MappingEntry(anilist_id=1)},
        )
        recording = install_recording_hub()
        _engine(_FinalizeRecorder(), logger, seadex=seadex).run_sync(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )
        return recording

    def test_outage_notes_unreachable_not_a_count(self, logger: logging.Logger) -> None:
        # The prefetch "return" is how many ids NEEDED fetching; on an outage
        # none were actually fetched, so the old "N entries" note was a lie.
        seadex = _FakeGateway()
        seadex.outage = True

        step = self._seadex_step(self._run(logger, seadex=seadex))

        assert step.detail == "unreachable"
        assert step.outcome is OutcomeCategory.DEFERRED  # graduates as a warning

    def test_healthy_prefetch_keeps_the_count_note(self, logger: logging.Logger) -> None:
        step = self._seadex_step(self._run(logger, seadex=_FakeGateway()))

        assert step.detail == "cached"  # the fake reports 0 fetched -> cache-warm
        assert step.outcome is OutcomeCategory.SUCCESS


class TestSelectionRecheck:
    """The stale-selection announcement + the full-coverage vouch rule.

    Vouch state is read back through `selection_stale`: after a vouch, any
    OTHER digest reads stale; with nothing vouched, everything reads fresh.
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
        process_returns: bool = False,
    ) -> tuple[RunLoop, RecordingHub]:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=3, title="A")],
            anilist_ids={11: MappingEntry(anilist_id=11)},
            process_returns=process_returns,
        )
        recording = install_recording_hub()
        engine = _engine(_FinalizeRecorder(), logger, config=config, seadex=seadex)
        engine._services._selection_stale = stale
        engine.run_sync(strategy, item_id=item_id, dry_run=True, boot=BootFlow())
        return engine, recording

    def _notes(self, recording: RecordingHub) -> list[str]:
        return [d.message for d in recording.of_type(Diagnostic)]

    def _vouched_any(self, engine: RunLoop) -> bool:
        """Whether the run recorded a digest at all (an alien digest then reads stale)."""

        return engine.cache_store.selection_stale(Arr.SONARR, "some-other-digest")

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
        # The ignore flag already re-checks everything; announcing a selection
        # re-check on top would be noise.
        _, recording = self._run(logger, stale=True, config=make_config(ignore_seadex_update_times=True))

        assert self._NOTE not in self._notes(recording)

    def test_full_run_vouches_the_current_digest(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger)

        assert engine.cache_store.selection_stale(Arr.SONARR, engine._config.selection_digest()) is False
        assert self._vouched_any(engine) is True

    def test_single_item_run_does_not_vouch(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger, item_id=3)

        assert self._vouched_any(engine) is False

    def test_capped_run_does_not_vouch(self, logger: logging.Logger) -> None:
        engine, _ = self._run(logger, process_returns=True)

        assert self._vouched_any(engine) is False

    def test_outage_run_does_not_vouch(self, logger: logging.Logger) -> None:
        outage = _FakeGateway()
        outage.outage = True
        engine, _ = self._run(logger, seadex=outage)

        assert self._vouched_any(engine) is False
