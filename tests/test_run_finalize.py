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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.cache import CacheStore
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import BootStepFinished, Diagnostic, Severity
from pearlarr.progress import ProgressCallback
from pearlarr.skyhook_gateway import SkyHookGateway
from pearlarr.tvdb_orders import AIRED_ORDER, SonarrSeriesOrders
from pearlarr.wait_types import Outcome, OutcomeKind
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    ScriptedAniListClient,
    drive_arr_run,
    grab_request,
    make_bare_instance,
    make_config,
    make_run_deps,
)
from .fakes import (
    FakeArrItem,
    FakeClock,
    FakeSkyHookClient,
    FakeStrategy,
    diagnostic_messages,
    install_recording_hub,
)
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

    `qbit` None is a run with no qBittorrent configured, so a dry run. AniList knows every title.
    """

    store = CacheStore.load(str(cache), config_hash="")
    try:
        client = None if qbit is None else qbit.client()
        deps = make_run_deps(cache_store=store, qbit=client, clock=clock, anilist_client=ScriptedAniListClient())
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

    @pytest.mark.parametrize(
        ("qbit", "next_fetch"),
        [pytest.param(FakeQbit(), "cached", id="real"), pytest.param(None, "4 titles", id="dry-run")],
    )
    def test_the_anilist_answers_the_prefetch_fetched_are_kept_before_the_scan(
        self, tmp_path: Path, qbit: FakeQbit | None, next_fetch: str
    ) -> None:
        # Stopped in the first title, before any library item commits. The next run fetches only what wasn't kept.
        cache = tmp_path / "cache.db"
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _Staging(services, stop_at=11), qbit=qbit)
        recording = install_recording_hub()

        _scan_into(cache, _Staging, qbit=FakeQbit())

        [step] = [step for step in recording.of_type(BootStepFinished) if step.label == "Fetching AniList metadata"]
        assert step.detail == next_fetch


type _Ending = Callable[[ArrRunServices, int], None]
"""How a title's check ends, given the run's services and the title's AniList id."""


def _grab(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends at the grab tail."""

    services.grab_and_cache(grab_request(anilist_id=anilist_id))


def _no_releases(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends with no release to grab."""

    services.report_no_releases(anilist_id, {})


def _invalid_pick(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends with an interactive pick that chose nothing valid."""

    del anilist_id
    services.report_invalid_selection()


def _no_query(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends having queried nothing, as one that skips an unchanged cached title does."""

    del services, anilist_id


class _Endings(FakeStrategy):
    """A library of one item whose titles 1, 2 and 3 are checked in turn, each ending as `endings` says."""

    def __init__(self, services: ArrRunServices, endings: tuple[_Ending, _Ending, _Ending]) -> None:
        super().__init__(items=[FakeArrItem()], anilist_ids={i: TitleMapping(anilist_id=i) for i in (1, 2, 3)})
        self._services = services
        self._endings = endings

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        self._endings[anilist_id - 1](self._services, anilist_id)


class TestPacingPause:
    """A check that queried owes the `sleep_time` pause, slept once before the next title."""

    @pytest.mark.parametrize(
        ("first", "sleeps"),
        [
            pytest.param(_grab, [3], id="grab"),
            pytest.param(_no_releases, [3], id="no-releases"),
            pytest.param(_invalid_pick, [3], id="invalid-pick"),
            pytest.param(_no_query, [], id="no-query"),
        ],
    )
    def test_a_check_that_queried_owes_one_pause_slept_before_the_next_title(
        self, first: _Ending, sleeps: list[float]
    ) -> None:
        # The two titles after the first query nothing, so the first's pause is all the run owes.
        clock = FakeClock()
        deps = make_run_deps(config=make_config(url="http://sonarr", api_key="key", sleep_time=3), clock=clock)

        drive_arr_run(deps, lambda services: _Endings(services, (first, _no_query, _no_query)))

        assert clock.sleeps == sleeps


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


_NOTED = OutcomeKind.IMPORTED
"""The outcome a boot step that finished with a note shows."""

_WARNED = OutcomeKind.LEFT
"""The outcome a boot step that finished with a warning shows."""


class TestBootNotes:
    """Each boot step ends on a note of what it fetched, or on a warning when its source was unreachable.

    An unreachable source's step never claims a count it didn't fetch.
    """

    @pytest.mark.parametrize(
        ("titles", "seadex_unreachable", "anilist_failing", "anilist_step", "seadex_step"),
        [
            pytest.param({1}, False, frozenset[int](), ("1 title", _NOTED), ("1 entry", _NOTED), id="healthy"),
            pytest.param(set[int](), False, frozenset[int](), ("cached", _NOTED), ("cached", _NOTED), id="no-titles"),
            pytest.param({1}, True, frozenset[int](), ("1 title", _NOTED), ("unreachable", _WARNED), id="no-seadex"),
            pytest.param({1}, False, frozenset({1}), ("unavailable", _WARNED), ("1 entry", _NOTED), id="no-anilist"),
        ],
    )
    def test_each_boot_step_notes_what_it_fetched_or_warns_that_its_source_was_unreachable(
        self,
        titles: set[int],
        seadex_unreachable: bool,
        anilist_failing: frozenset[int],
        anilist_step: tuple[str, OutcomeKind],
        seadex_step: tuple[str, OutcomeKind],
    ) -> None:
        # Both series map the same titles, so each prefetch fetches each title once.
        library = FakeStrategy(
            items=[FakeArrItem(item_id=1), FakeArrItem(item_id=2)],
            anilist_ids={anilist_id: TitleMapping(anilist_id=anilist_id) for anilist_id in titles},
        )
        deps = make_run_deps(
            seadex=FakeSeaDexSource(unreachable=seadex_unreachable),
            anilist_client=ScriptedAniListClient(failing=anilist_failing),
        )
        recording = install_recording_hub()

        drive_arr_run(deps, lambda _: library)

        assert [(step.label, step.detail, step.outcome) for step in recording.of_type(BootStepFinished)] == [
            ("Fetching Sonarr library", "2 series", _NOTED),
            ("Checking Sonarr activity", "none", _NOTED),
            ("Fetching AniList metadata", *anilist_step),
            ("Fetching SeaDex entries", *seadex_step),
        ]


# One import event on series 3, so the activity read has a checkpoint to stage.
_IMPORT_EVENT = HistoryRecord(
    id=1, date="2026-07-06T10:00:00Z", item_id=3, event_type="downloadFolderImported", download_id=None, reason=None
)

_OLDER_SELECTION = "an-older-selection"
"""The selection hash an earlier full run stored, under selection settings changed since."""


def _series_3(services: ArrRunServices | None = None) -> FakeStrategy:
    """A library of series 3 mapping title 11, its history the one import. `services` holds each title past the cap."""

    return FakeStrategy(
        items=[FakeArrItem(item_id=3)],
        anilist_ids={11: TitleMapping(anilist_id=11)},
        services=services,
        history=[_IMPORT_EVENT],
    )


def _left_for_the_next_run(deps: ArrRunDeps) -> tuple[bool, bool]:
    """Whether the run stored its selection settings and its activity checkpoint, both read for the next run."""

    cache = deps.cache_store
    return (
        not cache.selection_changed(Arr.SONARR, deps.config.selection_hash()),
        cache.get_history_checkpoint(Arr.SONARR) is not None,
    )


@dataclass(frozen=True, kw_only=True)
class _Situation:
    """What a run met: the series it was narrowed to, and which sources were unreachable or held a title."""

    item_id: int | None = None
    seadex_unreachable: bool = False
    anilist_unreachable: bool = False
    held_past_the_cap: bool = False


class TestSelectionRecheck:
    """A run over changed selection settings checks the cached titles again, and a full scan stores the new settings.

    The note says why the titles are checked again. A single-item run checks only its series and can't store the
    settings, so the note would overstate it and recur. With the ignore flag every title is checked anyway.
    """

    _NOTE = "Selection settings changed - checking cached titles again"

    @pytest.mark.parametrize(
        ("changed", "item_id", "ignore_update_times", "announced"),
        [
            pytest.param(True, None, False, True, id="changed"),
            pytest.param(False, None, False, False, id="unchanged"),
            pytest.param(True, 3, False, False, id="single-item"),
            pytest.param(True, None, True, False, id="ignoring-update-times"),
        ],
    )
    def test_a_changed_selection_is_announced_only_on_a_full_run_that_reads_update_times(
        self, changed: bool, item_id: int | None, ignore_update_times: bool, announced: bool
    ) -> None:
        config = make_config(url="http://sonarr", api_key="key", ignore_seadex_update_times=ignore_update_times)
        cache = FakeCacheStore()
        cache.store_selection_hash(Arr.SONARR, _OLDER_SELECTION if changed else config.selection_hash())
        recording = install_recording_hub()

        drive_arr_run(make_run_deps(config=config, cache_store=cache), lambda _: _series_3(), item_id=item_id)

        assert (self._NOTE in diagnostic_messages(recording)) is announced

    @pytest.mark.parametrize(
        ("situation", "stored"),
        [
            pytest.param(_Situation(), True, id="full"),
            pytest.param(_Situation(item_id=3), False, id="single-item"),
            pytest.param(_Situation(seadex_unreachable=True), False, id="no-seadex"),
            # A title AniList left unread isn't cached, so the next run checks it whatever this one stored.
            pytest.param(_Situation(anilist_unreachable=True), True, id="no-anilist"),
            # So is a title held past the cap.
            pytest.param(_Situation(held_past_the_cap=True), True, id="held-past-the-cap"),
        ],
    )
    def test_a_full_scan_seadex_answered_stores_the_selection_and_the_checkpoint(
        self, situation: _Situation, stored: bool
    ) -> None:
        # Left unstored, the next run checks the cached titles again and replays the same activity. The run is a real
        # one, as only a real run's stores are saved.
        cache = FakeCacheStore()
        cache.store_selection_hash(Arr.SONARR, _OLDER_SELECTION)
        deps = make_run_deps(
            cache_store=cache,
            qbit=FakeQbit().client(),
            seadex=FakeSeaDexSource(unreachable=situation.seadex_unreachable),
            anilist_client=ScriptedAniListClient(failing=frozenset({11} if situation.anilist_unreachable else ())),
        )

        drive_arr_run(
            deps,
            lambda services: _series_3(services if situation.held_past_the_cap else None),
            item_id=situation.item_id,
        )

        assert _left_for_the_next_run(deps) == (stored, stored)


class _OrderFetching(FakeStrategy):
    """Series 3 mapping title 11, its episodes warmed and its orders asked of `skyhook`, as Sonarr's series are."""

    def __init__(self, skyhook: SkyHookGateway) -> None:
        super().__init__(
            items=[FakeArrItem(item_id=3)], anilist_ids={11: TitleMapping(anilist_id=11)}, history=[_IMPORT_EVENT]
        )
        self._skyhook = skyhook

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_orders(self, items: list[FakeArrItem], *, progress: ProgressCallback | None = None) -> int:
        series = [SonarrSeriesOrders(item.id, AIRED_ORDER, None) for item in items]
        return self._skyhook.prefetch(series, dry_run=False, progress=progress)


@pytest.mark.parametrize(
    ("failing", "note", "full_scan"),
    [
        pytest.param(frozenset({(3, "official")}), "unreachable", False, id="outage"),
        pytest.param(frozenset[tuple[int, str]](), "1 series", True, id="healthy"),
    ],
)
def test_a_skyhook_unreachable_notes_unreachable_and_the_run_neither_stores_the_selection_hash_nor_commits(
    failing: frozenset[tuple[int, str]], note: str, full_scan: bool
) -> None:
    # A series SkyHook left unrefreshed was read by its old stored orders, so the run commits neither and the next one
    # repeats the re-checks a settings or file change drove. A real run, as only a real run's stores are saved.
    cache = FakeCacheStore()
    cache.store_selection_hash(Arr.SONARR, _OLDER_SELECTION)
    deps = make_run_deps(cache_store=cache, qbit=FakeQbit().client(), skyhook_client=FakeSkyHookClient(failing=failing))
    recording = install_recording_hub()

    drive_arr_run(deps, lambda _: _OrderFetching(deps.skyhook))

    [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching TVDB orders"]
    assert step.detail == note
    assert _left_for_the_next_run(deps) == (full_scan, full_scan)
