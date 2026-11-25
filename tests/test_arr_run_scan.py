# pyright: strict
# pyright: reportPrivateUsage=false
# Reads the loop's private run state and services. The repo disables reportPrivateUsage for tests, strict re-flags it.
"""Guards the arr run loop's scan: a title held past the cap, a failing title or item, and the item's own name."""

import logging
from typing import override

from pearlarr.boot_flow import BootFlow
from pearlarr.mappings import TitleMapping
from pearlarr.output import Diagnostic, Severity

from .fakes import FakeArrItem, FakeStrategy, install_recording_hub
from .run_loop_fakes import FinalizeRecorder, make_engine


class TestHeldTitleKeepsScanning:
    """A title held past the cap stops nothing: every item is scanned and the single finalize site runs once."""

    def test_held_title_keeps_scanning_and_finalizes_once(self, logger: logging.Logger) -> None:
        finalize = FinalizeRecorder()
        engine = make_engine(finalize, logger)
        # Every title's grab is held by the cap, as the pipeline holds a grabbable release past it.
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={1: TitleMapping(anilist_id=1)},
            services=engine._services,
        )

        engine.run_arr(strategy, item_id=None, dry_run=True, boot=BootFlow())

        # The hold on the first item never stopped the scan: the second item's title is still checked.
        assert strategy.process_calls == [1, 1]
        assert engine._services.state.stats.held_by_cap == 2
        # ...and the single post-loop finalize ran exactly once.
        assert finalize.calls == 1


class TestPerTitleErrorContainment:
    """One title's failure is contained to that title, not the whole item."""

    def test_one_title_error_does_not_stop_its_siblings(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: TitleMapping(anilist_id=1), 2: TitleMapping(anilist_id=2)},
            process_raises_on=1,
        )
        finalize = FinalizeRecorder()
        recording = install_recording_hub()
        make_engine(finalize, logger).run_arr(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # The first title raised but was contained: the sibling title 2 is still checked.
        assert strategy.process_calls == [1, 2]
        # The per-title failure is an ERROR Diagnostic with its traceback (containment
        # is observable).
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == "A (AniList #1): unexpected error (boom on anilist_id 1) - moving on to the next title"
        assert error.trace is not None
        # The single finalize still ran once on the normal path to the arr run's end.
        assert finalize.calls == 1


class _ItemRaisingStrategy(FakeStrategy):
    """A `FakeStrategy` whose mapping lookup raises for one item id.

    This is the hook `run_arr` hits per item BEFORE the per-title loop (the
    OUTER containment handler).
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
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        # Only the per-item scan calls with the log_ignored default. The pre-loop
        # activity/prefetch steps pass False and must stay healthy.
        if log_ignored and item.id == self._raise_on_item:
            raise RuntimeError("mapping lookup exploded")
        return super().item_mappings(item, log_ignored)


class TestPerItemErrorContainment:
    """A whole item's failure is contained to that item, not the arr run."""

    def test_one_item_error_does_not_stop_its_siblings(self, logger: logging.Logger) -> None:
        strategy = _ItemRaisingStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={5: TitleMapping(anilist_id=5)},
            raise_on_item=1,
        )
        finalize = FinalizeRecorder()
        recording = install_recording_hub()

        make_engine(finalize, logger).run_arr(
            strategy,
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        # Item A raised before its per-title loop began. Sibling item B was still scanned.
        assert strategy.process_calls == [5]
        # The per-item failure is an ERROR Diagnostic with its traceback.
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == "A: unexpected error (mapping lookup exploded) - moving on to the next library item"
        assert error.trace is not None
        # The arr run still reached the single post-loop finalize site.
        assert finalize.calls == 1


class TestScanItemState:
    """The scan puts the arr item's own name in the arr run state before any per-title line."""

    def test_arr_title_follows_the_item(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: TitleMapping(anilist_id=1)},
        )
        engine = make_engine(FinalizeRecorder(), logger)
        install_recording_hub()

        engine.run_arr(strategy, item_id=None, dry_run=True, boot=BootFlow())

        assert engine._state.arr_title == "A"
