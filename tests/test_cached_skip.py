# pyright: strict
# pyright: reportPrivateUsage=false
# These read the hub's private run context (run._ctx). Strict re-flags that and
# the repo disables reportPrivateUsage for tests.
"""The cached-title short-circuit (`ArrRunServices.is_unchanged`) and the title interfaces around it.

Pins the decision, made from a single `CacheStore.get_cached_title` read: a cached
title whose SeaDex `updated_at` still matches is unchanged (and its url/coverage
backfilled once if the row predates those fields, its name once AniList can
resolve it). An absent cached title, or one whose entry changed, is checked again,
and a row sent on to a check loses its stamp until a check caches the title.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, override

import httpx
import pytest
from seadex import EntryRecord

from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitle, CachedTitleUpdate
from pearlarr.config import AppConfig, Arr
from pearlarr.output import Accent, StyledValue, TitleDetail, TitleHeader
from pearlarr.output.events import TitleState
from pearlarr.pending_import import TitleNames
from pearlarr.tvdb_orders import AIRED_ORDER

from .builders import (
    AddStatus,
    FakeCacheStore,
    FakeSeaDexSource,
    FakeTorrents,
    ScriptedAniListClient,
    grab_request,
    make_anilist_gateway,
    make_config,
    make_entry_record,
    make_grab_pipeline,
    make_run_deps,
    make_services,
    one_release_dict,
)
from .fakes import RecordingHub, install_recording_hub
from .wait_fakes import FakeQbit

_URL = "https://releases.moe/7"
_COVERAGE = "S01 E01-E12"
_CACHED_ON = datetime(2026, 1, 1)
"""SeaDex's update time for title 7's entry, the one its row was cached under."""

_ENTRY = make_entry_record(anilist_id=7, url=_URL, updated_at=_CACHED_ON)

_SETTINGS = make_config()


def _checked() -> CachedTitleUpdate:
    """What a Sonarr check of title 7 gathers for its row, before the check's end adds the fallback mark.

    Fresh each call, since the end writes into it.
    """

    return {
        "updated_at": _CACHED_ON,
        "name": "Show",
        "url": _URL,
        "coverage": _COVERAGE,
        "season_type": AIRED_ORDER,
        "series_row_count": 12,
    }


def _cached_row(*, fallback_satisfied: bool = False) -> CachedTitleUpdate:
    """Title 7's row as a Sonarr check leaves it."""

    row = _checked()
    row["fallback_satisfied"] = fallback_satisfied
    return row


def _services(cache: FakeCacheStore, *, settings: AppConfig = _SETTINGS) -> ArrRunServices:
    """One run's services over the cache, built as bootstrap builds them for each run.

    qBittorrent is set up, so the run is no dry run and what it caches is what the next run reads.
    """

    deps = make_run_deps(config=settings, cache_store=cache, qbit=FakeQbit().client())
    return ArrRunServices(deps, Arr.SONARR)


def _holding(row: CachedTitleUpdate) -> FakeCacheStore:
    """A cache holding title 7's row."""

    cache = FakeCacheStore()
    cache.update_cached_title(Arr.SONARR, 7, row)
    return cache


def _shown_unchanged(recording: RecordingHub) -> list[int | None]:
    """The titles shown as unchanged, by AniList id."""

    return [header.anilist_id for header in recording.of_type(TitleHeader) if header.state is TitleState.UNCHANGED]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Gate:
    """What the gate weighs for title 7: its row, SeaDex's entry for it, and what the run knows."""

    row: CachedTitleUpdate | None = field(default_factory=_cached_row)
    """None for a title never cached."""
    entry: EntryRecord = _ENTRY
    settings: AppConfig = _SETTINGS
    last_scan: AppConfig | None = None
    """The settings the last full scan stored its selection hash under. None for the settings the run has."""
    files_changed: frozenset[int] = frozenset()
    """The titles whose files changed in Sonarr since the last run."""


def _run_under(gate: _Gate, cache: FakeCacheStore) -> ArrRunServices:
    """The run's services over the cache, once the cache and the run hold what the gate weighs."""

    cache.store_selection_hash(Arr.SONARR, (gate.last_scan or gate.settings).selection_hash())
    if gate.row is not None:
        cache.update_cached_title(Arr.SONARR, 7, gate.row)
    run = _services(cache, settings=gate.settings)
    run.mark_dirty(gate.files_changed)
    return run


_LEFT_ALONE: Final[tuple[bool, list[int | None]]] = (True, [7])
"""True to the strategy, which stops there, and an unchanged block shown for the title."""
_SENT_ON: Final[tuple[bool, list[int | None]]] = (False, [])
"""False, so the strategy checks the title, and nothing shown for it yet."""


class _RecordingCacheStore(FakeCacheStore):
    """The in-memory store, also recording each `update_cached_title` payload so one merged write can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[CachedTitleUpdate] = []

    @override
    def update_cached_title(self, arr: Arr, anilist_id: int, update: CachedTitleUpdate) -> None:
        self.updates.append(update)
        super().update_cached_title(arr, anilist_id, update)


class _RecordingReporter:
    """Records the title each `log_cached_title` call reports as cached."""

    def __init__(self) -> None:
        self.cached: list[int] = []

    def log_cached_title(self, state: ArrRunState, anilist_id: int) -> None:
        del state
        self.cached.append(anilist_id)


@dataclass(frozen=True, slots=True)
class _BackfillCase:
    """One cached-title backfill: the stored row, what AniList resolves, the single write, the wire calls, the name after."""

    stored: CachedTitleUpdate
    anilist: str | None
    updates: list[CachedTitleUpdate]
    queries: list[int]
    name: str | None


_FRESH = datetime(2021, 1, 1)


class TestLeavingATitleAlone:
    """A cached title is left alone only while nothing could change what its check would find."""

    @pytest.mark.parametrize(
        ("gate", "outcome"),
        [
            pytest.param(_Gate(), _LEFT_ALONE, id="cached under its entry's current update"),
            pytest.param(_Gate(row=None), _SENT_ON, id="never cached"),
            pytest.param(
                _Gate(entry=copy.replace(_ENTRY, updated_at=datetime(2026, 3, 1))),
                _SENT_ON,
                id="its entry updated since",
            ),
            pytest.param(
                _Gate(settings=make_config(ignore_seadex_update_times=True)),
                _SENT_ON,
                id="the settings ignore SeaDex's update times",
            ),
            pytest.param(_Gate(files_changed=frozenset({7})), _SENT_ON, id="its files changed"),
            pytest.param(_Gate(files_changed=frozenset({8})), _LEFT_ALONE, id="another title's files changed"),
            pytest.param(
                _Gate(last_scan=make_config(want_best=False)),
                _SENT_ON,
                id="the selection settings changed since the last full scan",
            ),
            pytest.param(
                _Gate(row=_cached_row(fallback_satisfied=True), settings=make_config(private_releases="warn")),
                _SENT_ON,
                id="a public fallback satisfied it and private releases warn",
            ),
            pytest.param(
                _Gate(row=_cached_row(fallback_satisfied=True), settings=make_config(private_releases="fallback")),
                _LEFT_ALONE,
                id="a public fallback satisfied it and private releases fall back",
            ),
        ],
    )
    def test_a_title_is_left_alone_only_while_nothing_could_change_its_check(
        self, gate: _Gate, outcome: tuple[bool, list[int | None]]
    ) -> None:
        recording = install_recording_hub()
        run = _run_under(gate, FakeCacheStore())

        left_alone = run.is_unchanged(7, gate.entry, lambda: _COVERAGE)

        assert (left_alone, _shown_unchanged(recording)) == outcome


class TestIsUnchanged:
    """An unchanged title backfills the url/coverage and the name its row lacks, once."""

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                _BackfillCase(
                    stored={"updated_at": _FRESH},
                    anilist=None,
                    updates=[{"url": "sd-url", "coverage": "S01"}],
                    queries=[7],
                    name=None,
                ),
                id="url and coverage a legacy row lacks",
            ),
            pytest.param(
                _BackfillCase(
                    stored={"url": "u", "updated_at": _FRESH},
                    anilist="Resolved",
                    updates=[{"name": "Resolved"}],
                    queries=[7],
                    name="Resolved",
                ),
                id="name a row written during an outage lacks",
            ),
            pytest.param(
                _BackfillCase(
                    stored={"updated_at": _FRESH},
                    anilist="Resolved",
                    updates=[{"url": "sd-url", "coverage": "S01", "name": "Resolved"}],
                    queries=[7],
                    name="Resolved",
                ),
                id="both merged into one write",
            ),
            pytest.param(
                _BackfillCase(
                    stored={"name": "Kept", "url": "u", "updated_at": _FRESH},
                    anilist="Resolved",
                    updates=[],
                    queries=[],
                    name="Kept",
                ),
                id="a stored name spares the lookup",
            ),
            pytest.param(
                _BackfillCase(
                    stored={"url": "u", "updated_at": _FRESH},
                    anilist=None,
                    updates=[],
                    queries=[7],
                    name=None,
                ),
                id="nothing to resolve writes nothing",
            ),
        ],
    )
    def test_backfills_once_in_a_single_write(self, case: _BackfillCase) -> None:
        # Nothing else checks a fresh cached title again, so the unchanged check is where an older row
        # picks up its url/coverage and a row written while AniList was unreachable picks up its name.
        cache = _RecordingCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, case.stored)
        cache.updates.clear()
        client = ScriptedAniListClient(case.anilist)
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _anilist=make_anilist_gateway(client))

        assert run.is_unchanged(7, make_entry_record(updated_at=_FRESH, url="sd-url"), lambda: "S01") is True

        assert cache.updates == case.updates
        assert client.query_calls == case.queries
        persisted = cache.get_cached_title(Arr.SONARR, 7)
        assert persisted is not None
        assert persisted.name == case.name


class TestACheckedTitleLosesItsStamp:
    """The gate clears the stamp of a title it sends on to a check: only a check that caches it writes it back."""

    _ENTRY = make_entry_record(updated_at=_CACHED_ON)

    @staticmethod
    def _run(cache: FakeCacheStore, **wiring: object) -> ArrRunServices:
        return make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
            **wiring,
        )

    def _rechecking(self, cache: FakeCacheStore, torrents: FakeTorrents, *, add_cap: int = 0) -> ArrRunServices:
        """A run made to check title 7 again over its cached row, as a library change makes it."""

        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": _CACHED_ON})
        pipeline = make_grab_pipeline(_torrents=torrents, cache_store=cache, sleep_time=0, max_torrents_to_add=add_cap)
        pipeline._anilist.answers.update({7: {}})
        pipeline._state.per_title.title_name = "Show"
        run = self._run(cache, _grab_pipeline=pipeline, _state=pipeline._state)
        run.mark_dirty([7])
        return run

    def _grab(self, run: ArrRunServices) -> None:
        """The check's tail over one planned grab, hash h1, set up as an arr strategy sets it up."""

        run.grab_and_cache(
            grab_request(
                anilist_id=7,
                entry=self._ENTRY,
                seadex_dict=one_release_dict(srg="RG", infohash="h1"),
                update=run.new_cached_title_update(run.resolve_title(7), self._ENTRY),
            )
        )

    def test_a_title_sent_on_to_a_check_loses_its_stamp(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": _CACHED_ON})
        run = self._run(cache)
        run.mark_dirty([7])

        assert run.is_unchanged(7, self._ENTRY, lambda: "") is False

        cached = cache.get_cached_title(Arr.SONARR, 7)
        assert cached is not None
        assert cached.updated_at is None

    def test_an_unchanged_title_keeps_its_stamp(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": _CACHED_ON})

        assert self._run(cache).is_unchanged(7, self._ENTRY, lambda: "") is True
        assert self._run(cache).is_unchanged(7, self._ENTRY, lambda: "") is True

    def test_a_recheck_whose_grab_fails_is_checked_again_next_run(self) -> None:
        # The row matched SeaDex's current update before the re-check, and nothing forces the next run.
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")}))

        assert run.is_unchanged(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert self._run(cache).is_unchanged(7, self._ENTRY, lambda: "") is False

    def test_a_recheck_held_past_the_add_cap_is_checked_again_next_run(self) -> None:
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), add_cap=1)
        run.state.torrents_added = 1

        assert run.is_unchanged(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert run.state.stats.held_by_cap == 1
        assert self._run(cache).is_unchanged(7, self._ENTRY, lambda: "") is False

    def test_a_recheck_that_caches_the_title_is_unchanged_next_run(self) -> None:
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}))

        assert run.is_unchanged(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert self._run(cache).is_unchanged(7, self._ENTRY, lambda: "") is True


class _TailReporter:
    """Records the no-releases / no-entry / SeaDex-unreachable outcomes the shared tails report."""

    def __init__(self) -> None:
        self.no_releases_ctxs: list[ArrRunState] = []
        self.no_sd_entry_ids: list[int] = []
        self.seadex_unreachable_ids: list[int] = []

    def log_no_seadex_releases(self, state: ArrRunState) -> bool:
        self.no_releases_ctxs.append(state)
        return True

    def log_no_sd_entry(self, state: ArrRunState, anilist_id: int) -> bool:
        del state
        self.no_sd_entry_ids.append(anilist_id)
        return True

    def log_seadex_unreachable(self, state: ArrRunState, anilist_id: int) -> bool:
        del state
        self.seadex_unreachable_ids.append(anilist_id)
        return True


_NO_RELEASES = TitleDetail("status", StyledValue("no suitable releases on SeaDex", Accent.DIM))


class TestATitleWithNoReleases:
    """A check that finds no suitable releases on SeaDex says so, and caches what it found unless it was read short."""

    def test_it_is_shown_and_cached_with_what_its_check_found(self) -> None:
        recording = install_recording_hub()
        cache = FakeCacheStore()

        _services(cache).report_no_releases(7, _checked())

        assert recording.of_type(TitleDetail) == [_NO_RELEASES]
        assert cache.get_cached_title(Arr.SONARR, 7) == CachedTitle(
            updated_at="2026-01-01 00:00:00",
            name="Show",
            url=_URL,
            coverage=_COVERAGE,
            fallback_satisfied=False,
            season_type=AIRED_ORDER,
            series_row_count=12,
        )

    def test_it_loses_the_mark_a_public_fallback_left(self) -> None:
        # The write merges into the row, so the mark an earlier fallback grab left stays unless this end clears it.
        cache = _holding(_cached_row(fallback_satisfied=True))

        _services(cache).report_no_releases(7, _checked())

        row = cache.get_cached_title(Arr.SONARR, 7)
        assert row is not None
        assert row.fallback_satisfied is False

    def test_a_title_read_without_its_metadata_is_logged_and_left_uncached(self) -> None:
        recording = install_recording_hub()
        cache = FakeCacheStore()

        _services(cache).report_no_releases(7, _checked(), metadata_unread=True)

        assert recording.of_type(TitleDetail) == [_NO_RELEASES]
        assert cache.get_cached_title(Arr.SONARR, 7) is None


class TestBeginCheck:
    """`begin_check` reports no entry vs an unreachable SeaDex distinctly, and resets skip flags/tallies on a hit."""

    def test_no_seadex_entry_reports_and_returns_none(self) -> None:
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(), _reporter=reporter)

        assert run.begin_check(5) is None
        assert reporter.no_sd_entry_ids == [5]
        assert reporter.seadex_unreachable_ids == []
        assert run._state.stats.checked == 1

    def test_seadex_unreachable_reports_distinctly_and_returns_none(self) -> None:
        # A SeaDex-unreachable title must never be reported as "no entry" - it
        # takes the unreachable reporter hook and its own tally.
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(unreachable=True), _reporter=reporter)

        assert run.begin_check(5) is None
        assert reporter.seadex_unreachable_ids == [5]
        assert reporter.no_sd_entry_ids == []
        assert run._state.stats.checked == 1

    def test_entry_found_resets_skip_flags_and_tallies(self) -> None:
        entry = make_entry_record()
        run = make_services(_seadex=FakeSeaDexSource({5: entry}), _reporter=_TailReporter())
        run._state.per_title.private_only_skipped = True  # leftover flag from a previous title

        assert run.begin_check(5) is entry
        assert run._state.per_title.private_only_skipped is False
        assert run._state.stats.checked == 1


class TestResolveTitle:
    """`resolve_title` shows AniList's name, else the arr's, else the id form."""

    @staticmethod
    def _run(client: ScriptedAniListClient, arr_title: str = "") -> ArrRunServices:
        run = make_services(_anilist=make_anilist_gateway(client))
        run._state.arr_title = arr_title
        return run

    @pytest.mark.parametrize(
        ("anilist", "arr_title", "display"),
        [
            pytest.param("Resolved", "Series", "Resolved", id="anilist title"),
            pytest.param(None, "Series", "Series", id="arr title"),
            pytest.param(None, "", "AniList #5", id="id form"),
        ],
    )
    def test_climbs_the_ladder(self, anilist: str | None, arr_title: str, display: str) -> None:
        run = self._run(ScriptedAniListClient(anilist), arr_title=arr_title)

        assert run.resolve_title(5) == Title(
            anilist_id=5, name=display, names=TitleNames(arr_title, (anilist,) if anilist else ())
        )


class TestNewCacheDetails:
    """`new_cached_title_update` builds the cached title update, carrying a name only when AniList resolved one."""

    @pytest.mark.parametrize(
        ("anilist", "name"),
        [
            pytest.param("Resolved", {"name": "Resolved"}, id="resolved title rides as the name"),
            pytest.param(None, {}, id="fallback title is never stored"),
        ],
    )
    def test_carries_the_name_only_when_anilist_resolved_it(self, anilist: str | None, name: CachedTitleUpdate) -> None:
        # A fallback label must not clobber a name a prior run stored (the merge
        # keeps the omitted field), nor persist an id form a later run would have to undo.
        entry = make_entry_record(updated_at=_FRESH)
        run = make_services()

        title = Title(anilist_id=5, name="Series", names=TitleNames("Series", (anilist,) if anilist else ()))

        details = run.new_cached_title_update(title, entry)

        assert details == {**name, "updated_at": entry.updated_at}
