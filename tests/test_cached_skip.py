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

from dataclasses import dataclass
from datetime import datetime
from typing import override

import httpx
import pytest

from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import Arr
from pearlarr.pending_import import TitleNames

from .builders import (
    AddStatus,
    FakeCacheStore,
    FakeSeaDexSource,
    FakeTorrents,
    ScriptedAniListClient,
    grab_request,
    make_anilist_gateway,
    make_entry_record,
    make_grab_pipeline,
    make_services,
    one_release_dict,
)


class _RecordingCacheStore(FakeCacheStore):
    """The in-memory store, also recording each `update_cached_title` payload so one merged write can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[CachedTitleUpdate] = []

    @override
    def update_cached_title(self, arr: Arr, al_id: int, update: CachedTitleUpdate) -> None:
        self.updates.append(update)
        super().update_cached_title(arr, al_id, update)


class _RecordingReporter:
    """Records the title each `log_cached_title` call reports as cached."""

    def __init__(self) -> None:
        self.cached: list[int] = []

    def log_cached_title(self, state: ArrRunState, al_id: int) -> None:
        del state
        self.cached.append(al_id)


@dataclass(frozen=True, slots=True)
class _BackfillCase:
    """One cached-title backfill: the stored row, what AniList resolves, the single write, the wire calls, the name after."""

    stored: CachedTitleUpdate
    anilist: str | None
    updates: list[CachedTitleUpdate]
    queries: list[int]
    name: str | None


_FRESH = datetime(2021, 1, 1)


class TestIsUnchanged:
    """`is_unchanged` is True only for a fresh cached title, backfilling url/coverage and the name once.

    It is bypassed by `ignore_seadex_update_times` or a dirty id, and the dirty set clears each run.
    """

    @staticmethod
    def _run(cache: FakeCacheStore) -> ArrRunServices:
        # is_unchanged touches only cache_store + _config (real, default
        # ignore_seadex_update_times=False) + the reporter + the AniList gateway the
        # name backfill asks. ctx defaults to a SONARR ArrRunState (make_services).
        return make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
        )

    def test_unchanged_when_cached_and_timestamp_matches(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        reporter = _RecordingReporter()
        run = make_services(
            cache_store=cache,
            _reporter=reporter,
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
        )
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True
        assert reporter.cached == [7]

    def test_checks_when_the_cached_title_is_absent(self) -> None:
        run = self._run(FakeCacheStore())
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_checks_when_the_timestamp_changed(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        # SeaDex entry now carries a newer updated_at -> changed -> check again.
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2022, 6, 6)), lambda: "") is False

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

    def test_ignore_update_times_reprocesses_even_when_fresh(self) -> None:
        # The config escape hatch: a fresh, matching timestamp is still re-processed.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            ignore_seadex_update_times=True,
        )
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_dirty_id_reprocesses_even_when_fresh(self) -> None:
        # An arr-side file change bypasses the unchanged check despite a matching timestamp.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([7])
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_selection_changed_reprocesses_even_when_fresh(self) -> None:
        # A matching-preference change bypasses the unchanged check despite a matching timestamp.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _selection_changed=True)
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_non_dirty_sibling_stays_unchanged(self) -> None:
        # Marking one id dirty must not widen the bypass to other cached ids.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([8])
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

    def test_begin_run_clears_the_dirty_set(self) -> None:
        # Dirty ids are per-run state: the next run's rebind must reset them.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
            _filter=_CtxBind(),
            _grab_pipeline=_CtxBind(),
            _pending_imports=_CtxBind(),
        )
        run.mark_dirty([7])
        run.bind_arr_run(run.state)
        assert run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True


_CACHED_ON = datetime(2026, 1, 1)
"""SeaDex's update time for title 7's entry, the stamp its row carries while it is cached."""


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
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show"
        run = self._run(cache, _grab_pipeline=pipeline, _state=pipeline._state)
        run.mark_dirty([7])
        return run

    def _grab(self, run: ArrRunServices) -> None:
        """The check's tail over one planned grab, hash h1, set up as an arr strategy sets it up."""

        run.grab_and_cache(
            grab_request(
                al_id=7,
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


class TestFallbackSatisfiedResurfacing:
    """The fallback-satisfied marker bypasses the unchanged check in warn mode only."""

    @staticmethod
    def _cache(*, marker: bool) -> FakeCacheStore:
        cache = FakeCacheStore()
        cache.update_cached_title(
            Arr.SONARR,
            7,
            {"url": "u", "updated_at": datetime(2021, 1, 1), "fallback_satisfied": marker},
        )
        return cache

    @staticmethod
    def _skips(cache: FakeCacheStore, private_releases: str) -> bool:
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
            private_releases=private_releases,
        )
        return run.is_unchanged(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "")

    def test_warn_mode_reprocesses_a_marked_title(self) -> None:
        # Fresh timestamp, but the title was satisfied by a fallback: warn mode
        # checks it again so the private-only warning resurfaces.
        assert self._skips(self._cache(marker=True), "warn") is False

    def test_fallback_mode_still_finds_a_marked_title_unchanged(self) -> None:
        assert self._skips(self._cache(marker=True), "fallback") is True

    def test_warn_mode_finds_an_unmarked_title_unchanged(self) -> None:
        assert self._skips(self._cache(marker=False), "warn") is True

    def test_report_no_releases_clears_a_preseeded_marker(self) -> None:
        # A title that stops yielding usable releases is never fallback-satisfied:
        # the shared tail overwrites an old True.
        cache = self._cache(marker=True)
        run = make_services(cache_store=cache, _reporter=_TailReporter(), sleep_time=0)
        run.report_no_releases(7, {"name": "Title"})
        cached = cache.get_cached_title(Arr.SONARR, 7)
        assert cached is not None
        assert cached.fallback_satisfied is False


class _CtxBind:
    """A no-op ctx-bind collaborator for driving `begin_run` on a bare hub."""

    def bind_arr_run(self, state: ArrRunState) -> None:
        del state


class _TailReporter:
    """Records the no-releases / no-entry / SeaDex-unreachable outcomes the shared tails report."""

    def __init__(self) -> None:
        self.no_releases_ctxs: list[ArrRunState] = []
        self.no_sd_entry_ids: list[int] = []
        self.seadex_unreachable_ids: list[int] = []

    def log_no_seadex_releases(self, state: ArrRunState) -> bool:
        self.no_releases_ctxs.append(state)
        return True

    def log_no_sd_entry(self, state: ArrRunState, al_id: int) -> bool:
        del state
        self.no_sd_entry_ids.append(al_id)
        return True

    def log_seadex_unreachable(self, state: ArrRunState, al_id: int) -> bool:
        del state
        self.seadex_unreachable_ids.append(al_id)
        return True


class TestReportNoReleases:
    """`report_no_releases` runs the real four-step tail: log, cache write, throttle, then report not-grabbed."""

    def test_logs_persists_and_reports_not_grabbed(self) -> None:
        # The real three-step tail (log + cache write + throttle). Its body
        # previously had fake-only coverage: the interface tests script it, nothing
        # drove the real thing.
        cache = FakeCacheStore()
        reporter = _TailReporter()
        run = make_services(cache_store=cache, _reporter=reporter, sleep_time=0)

        run.report_no_releases(7, {"name": "Title", "url": "u"})

        persisted = cache.get_cached_title(Arr.SONARR, 7)
        assert persisted is not None
        assert (persisted.name, persisted.url) == ("Title", "u")
        assert len(reporter.no_releases_ctxs) == 1

    def test_a_title_read_without_its_metadata_is_logged_and_left_uncached(self) -> None:
        cache = FakeCacheStore()
        reporter = _TailReporter()
        run = make_services(cache_store=cache, _reporter=reporter, sleep_time=0)

        run.report_no_releases(7, {"name": "Title", "url": "u"}, metadata_unread=True)

        assert cache.get_cached_title(Arr.SONARR, 7) is None
        assert len(reporter.no_releases_ctxs) == 1


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
            al_id=5, name=display, names=TitleNames(arr_title, (anilist,) if anilist else ())
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

        title = Title(al_id=5, name="Series", names=TitleNames("Series", (anilist,) if anilist else ()))

        details = run.new_cached_title_update(title, entry)

        assert details == {**name, "updated_at": entry.updated_at}
