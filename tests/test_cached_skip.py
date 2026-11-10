# pyright: strict
# pyright: reportPrivateUsage=false
# These read the hub's private run context (run._ctx). Strict re-flags that and
# the repo disables reportPrivateUsage for tests.
"""The cached-entry short-circuit (`RunServices.cached_entry_skip`) and the title seams around it.

Pins the skip decision, made from a single `CacheStore.get_entry` read: a cached
entry whose SeaDex `updated_at` still matches is skipped (and its url/coverage
backfilled once if the record predates those fields, its name once AniList can
resolve it). An absent or stale entry is re-processed, and a row sent on to a
check loses its stamp until a check caches the title as done.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import override

import httpx
import pytest

from pearlarr.cache import CacheRecord
from pearlarr.config import Arr
from pearlarr.manual_import import EntryNames
from pearlarr.reporter import EntryTitle, RunContext
from pearlarr.run_services import RunServices

from .builders import (
    AddOutcome,
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
    """The in-memory store, also recording each `update_cache` payload so one merged write can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[CacheRecord] = []

    @override
    def update_cache(self, arr: Arr, al_id: int, cache_details: CacheRecord) -> None:
        self.updates.append(cache_details)
        super().update_cache(arr, al_id, cache_details)


class _RecordingReporter:
    """Records the title each `log_cached_entry` call reports as cached."""

    def __init__(self) -> None:
        self.cached: list[int] = []

    def log_cached_entry(self, ctx: RunContext, al_id: int) -> None:
        del ctx
        self.cached.append(al_id)


@dataclass(frozen=True, slots=True)
class _BackfillCase:
    """One cached-entry backfill: the stored row, what AniList resolves, the single write, the wire calls, the name after."""

    stored: CacheRecord
    anilist: str | None
    updates: list[CacheRecord]
    queries: list[int]
    name: str | None


_FRESH = datetime(2021, 1, 1)


class TestCachedEntrySkip:
    """`cached_entry_skip` skips only a fresh cached entry, backfilling url/coverage and the name once.

    It is bypassed by `ignore_seadex_update_times` or a dirty id, and the dirty set clears each run.
    """

    @staticmethod
    def _run(cache: FakeCacheStore) -> RunServices:
        # cached_entry_skip touches only cache_store + _config (real, default
        # ignore_seadex_update_times=False) + the reporter + the AniList gateway the
        # name backfill asks. ctx defaults to a SONARR RunContext (make_services).
        return make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
        )

    def test_skips_when_cached_and_timestamp_matches(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        reporter = _RecordingReporter()
        run = make_services(
            cache_store=cache,
            _reporter=reporter,
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
        )
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True
        assert reporter.cached == [7]

    def test_does_not_skip_when_entry_absent(self) -> None:
        run = self._run(FakeCacheStore())
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_does_not_skip_when_timestamp_is_stale(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        # SeaDex entry now carries a newer updated_at -> stale -> re-process.
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2022, 6, 6)), lambda: "") is False

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
        # Nothing else re-processes a fresh entry, so the skip is where an older record
        # picks up its url/coverage and a record written during an outage picks up its name.
        cache = _RecordingCacheStore()
        cache.update_cache(Arr.SONARR, 7, case.stored)
        cache.updates.clear()
        client = ScriptedAniListClient(case.anilist)
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _anilist=make_anilist_gateway(client))

        assert run.cached_entry_skip(7, make_entry_record(updated_at=_FRESH, url="sd-url"), lambda: "S01") is True

        assert cache.updates == case.updates
        assert client.query_calls == case.queries
        persisted = cache.get_entry(Arr.SONARR, 7)
        assert persisted is not None
        assert persisted.name == case.name

    def test_ignore_update_times_reprocesses_even_when_fresh(self) -> None:
        # The config escape hatch: a fresh, matching timestamp is still re-processed.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            ignore_seadex_update_times=True,
        )
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_dirty_id_reprocesses_even_when_fresh(self) -> None:
        # An arr-side file change bypasses the skip despite a matching timestamp.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([7])
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_selection_stale_reprocesses_even_when_fresh(self) -> None:
        # A matching-preference change bypasses the skip despite a matching timestamp.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _selection_stale=True)
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_non_dirty_sibling_still_skips(self) -> None:
        # Marking one id dirty must not widen the bypass to other cached ids.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([8])
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

    def test_begin_run_clears_the_dirty_set(self) -> None:
        # Dirty ids are per-run state: the next run's rebind must reset them.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
            _filter=_CtxBind(),
            _grab_pipeline=_CtxBind(),
            _records=_CtxBind(),
        )
        run.mark_dirty([7])
        run.begin_run(run.ctx)
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True


_CACHED_ON = datetime(2026, 1, 1)
"""SeaDex's update time for entry 7, the stamp its row carries while it is cached as done."""


class TestACheckedTitleLosesItsStamp:
    """The gate clears the stamp of a title it sends on to a check: only a check that caches it as done writes it back."""

    _ENTRY = make_entry_record(updated_at=_CACHED_ON)

    @staticmethod
    def _run(cache: FakeCacheStore, **wiring: object) -> RunServices:
        return make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=make_anilist_gateway(ScriptedAniListClient(None)),
            **wiring,
        )

    def _rechecking(self, cache: FakeCacheStore, torrents: FakeTorrents, *, add_cap: int = 0) -> RunServices:
        """A run made to check entry 7 again over its cached row, as a library change makes it."""

        cache.update_cache(Arr.SONARR, 7, {"updated_at": _CACHED_ON})
        pipeline = make_grab_pipeline(_torrents=torrents, cache_store=cache, sleep_time=0, max_torrents_to_add=add_cap)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show"
        run = self._run(cache, _grab_pipeline=pipeline, _ctx=pipeline._ctx)
        run.mark_dirty([7])
        return run

    def _grab(self, run: RunServices) -> None:
        """The check's tail over one flagged release, hash h1, seeded as a strategy seeds it."""

        run.grab_and_cache(
            grab_request(
                al_id=7,
                entry=self._ENTRY,
                seadex_dict=one_release_dict(srg="RG", infohash="h1"),
                cache_details=run.new_cache_details(run.resolve_title(7), self._ENTRY),
            )
        )

    def test_a_title_sent_on_to_a_check_loses_its_stamp(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": _CACHED_ON})
        run = self._run(cache)
        run.mark_dirty([7])

        assert run.cached_entry_skip(7, self._ENTRY, lambda: "") is False

        entry = cache.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.updated_at is None

    def test_a_skipped_title_keeps_its_stamp(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": _CACHED_ON})

        assert self._run(cache).cached_entry_skip(7, self._ENTRY, lambda: "") is True
        assert self._run(cache).cached_entry_skip(7, self._ENTRY, lambda: "") is True

    def test_a_recheck_whose_grab_fails_is_checked_again_next_run(self) -> None:
        # The row matched SeaDex's current update before the re-check, and nothing forces the next run.
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")}))

        assert run.cached_entry_skip(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert self._run(cache).cached_entry_skip(7, self._ENTRY, lambda: "") is False

    def test_a_recheck_held_past_the_add_cap_is_checked_again_next_run(self) -> None:
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")}), add_cap=1)
        run.ctx.torrents_added = 1

        assert run.cached_entry_skip(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert run.ctx.stats.held_by_cap == 1
        assert self._run(cache).cached_entry_skip(7, self._ENTRY, lambda: "") is False

    def test_a_recheck_that_caches_the_title_as_done_is_skipped_next_run(self) -> None:
        cache = FakeCacheStore()
        run = self._rechecking(cache, FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")}))

        assert run.cached_entry_skip(7, self._ENTRY, lambda: "") is False
        self._grab(run)

        assert self._run(cache).cached_entry_skip(7, self._ENTRY, lambda: "") is True


class TestFallbackSatisfiedResurfacing:
    """The fallback-satisfied marker bypasses the cache skip in warn mode only."""

    @staticmethod
    def _cache(*, marker: bool) -> FakeCacheStore:
        cache = FakeCacheStore()
        cache.update_cache(
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
        return run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "")

    def test_warn_mode_reprocesses_a_marked_entry(self) -> None:
        # Fresh timestamp, but the title was satisfied by a fallback: warn mode
        # re-processes it so the private-only warning resurfaces.
        assert self._skips(self._cache(marker=True), "warn") is False

    def test_fallback_mode_still_skips_a_marked_entry(self) -> None:
        assert self._skips(self._cache(marker=True), "fallback") is True

    def test_warn_mode_skips_an_unmarked_entry(self) -> None:
        assert self._skips(self._cache(marker=False), "warn") is True

    def test_no_releases_skip_clears_a_preseeded_marker(self) -> None:
        # A title that stops yielding usable releases is never fallback-satisfied:
        # the shared tail overwrites a stale True.
        cache = self._cache(marker=True)
        run = make_services(cache_store=cache, _reporter=_TailReporter(), sleep_time=0)
        run.no_releases_skip(7, {"name": "Title"})
        entry = cache.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is False


class _CtxBind:
    """A no-op ctx-bind collaborator for driving `begin_run` on a bare hub."""

    def begin_run(self, ctx: RunContext) -> None:
        del ctx


class _TailReporter:
    """Records the no-releases / no-entry / outage-skip outcomes the shared tails report."""

    def __init__(self) -> None:
        self.no_releases_ctxs: list[RunContext] = []
        self.no_sd_entry_ids: list[int] = []
        self.outage_skip_ids: list[int] = []

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        self.no_releases_ctxs.append(ctx)
        return True

    def log_no_sd_entry(self, ctx: RunContext, al_id: int) -> bool:
        del ctx
        self.no_sd_entry_ids.append(al_id)
        return True

    def log_seadex_outage_skip(self, ctx: RunContext, al_id: int) -> bool:
        del ctx
        self.outage_skip_ids.append(al_id)
        return True


class TestNoReleasesSkip:
    """`no_releases_skip` runs the real four-step tail: log, cache write, throttle, then report not-grabbed."""

    def test_logs_persists_and_reports_not_grabbed(self) -> None:
        # The real three-step tail (log + cache write + throttle). Its body
        # previously had fake-only coverage: the seam tests script it, nothing
        # drove the real thing.
        cache = FakeCacheStore()
        reporter = _TailReporter()
        run = make_services(cache_store=cache, _reporter=reporter, sleep_time=0)

        run.no_releases_skip(7, {"name": "Title", "url": "u"})

        persisted = cache.get_entry(Arr.SONARR, 7)
        assert persisted is not None
        assert (persisted.name, persisted.url) == ("Title", "u")
        assert len(reporter.no_releases_ctxs) == 1


class TestAlIdPrologue:
    """`al_id_prologue` reports a missing entry vs an outage distinctly, and resets skip flags/tallies on a hit."""

    def test_no_seadex_entry_reports_and_returns_none(self) -> None:
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(), _reporter=reporter)

        assert run.al_id_prologue(5) is None
        assert reporter.no_sd_entry_ids == [5]
        assert reporter.outage_skip_ids == []
        assert run._ctx.stats.checked == 1

    def test_outage_skip_reports_distinctly_and_returns_none(self) -> None:
        # A SeaDex-unreachable skip must never be reported as "no entry" - it
        # takes the outage reporter hook and its own tally.
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(outage=True), _reporter=reporter)

        assert run.al_id_prologue(5) is None
        assert reporter.outage_skip_ids == [5]
        assert reporter.no_sd_entry_ids == []
        assert run._ctx.stats.checked == 1

    def test_entry_found_resets_skip_flags_and_tallies(self) -> None:
        entry = make_entry_record()
        run = make_services(_seadex=FakeSeaDexSource({5: entry}), _reporter=_TailReporter())
        run._ctx.per_title.private_only_skipped = True  # stale flag from a previous title

        assert run.al_id_prologue(5) is entry
        assert run._ctx.per_title.private_only_skipped is False
        assert run._ctx.stats.checked == 1


class TestResolveTitle:
    """`resolve_title` shows AniList's title, else the arr's own, else the id form."""

    @staticmethod
    def _run(client: ScriptedAniListClient, arr_title: str = "") -> RunServices:
        run = make_services(_anilist=make_anilist_gateway(client))
        run._ctx.arr_title = arr_title
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

        assert run.resolve_title(5) == EntryTitle(
            al_id=5, display=display, names=EntryNames(arr_title, (anilist,) if anilist else ())
        )


class TestNewCacheDetails:
    """`new_cache_details` seeds the record, carrying a name only when AniList resolved one."""

    @pytest.mark.parametrize(
        ("anilist", "name"),
        [
            pytest.param("Resolved", {"name": "Resolved"}, id="resolved title rides as the name"),
            pytest.param(None, {}, id="fallback title is never stored"),
        ],
    )
    def test_carries_the_name_only_when_anilist_resolved_it(self, anilist: str | None, name: CacheRecord) -> None:
        # A fallback label must not clobber a name a prior run stored (the merge
        # keeps the omitted field), nor persist an id form a later run would have to undo.
        entry = make_entry_record(updated_at=_FRESH)
        run = make_services()

        title = EntryTitle(al_id=5, display="Series", names=EntryNames("Series", (anilist,) if anilist else ()))

        details = run.new_cache_details(title, entry)

        assert details == {**name, "updated_at": entry.updated_at}
