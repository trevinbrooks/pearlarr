# pyright: strict
# pyright: reportPrivateUsage=false
# These read the hub's private run context (run._ctx). Strict re-flags that and
# the repo disables reportPrivateUsage for tests.
"""The cached-entry short-circuit (`RunServices.cached_entry_skip`) and the title seams around it.

Pins the skip decision after it was folded onto a single `CacheStore.get_entry`
read (was a `check_al_id_in_cache` + a per-field `get_cached_field`): a cached
entry whose SeaDex `updated_at` still matches is skipped (and its url/coverage
backfilled once if the record predates those fields, its name once AniList can
resolve it). An absent or stale entry is re-processed.
"""

from datetime import datetime
from typing import Any, override

import httpx

from pearlarr.anilist_client import AniListClient
from pearlarr.anilist_gateway import AniListGateway
from pearlarr.cache import CacheRecord
from pearlarr.config import Arr
from pearlarr.log import EntryState
from pearlarr.reporter import RunContext
from pearlarr.run_services import EntryTitle, RunServices

from .builders import FakeCacheStore, FakeSeaDexSource, make_entry_record, make_logger, make_services


class _ScriptedTitleClient(AniListClient):
    """Scripted AniList wire client: a fixed resolvable title or none, queries recorded."""

    def __init__(self, title: str | None) -> None:
        super().__init__(client=httpx.Client())
        self._title = title
        self.query_calls: list[int] = []

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        self.query_calls.append(al_id)
        if self._title is None:
            return {}
        return {"data": {"Media": {"id": al_id, "title": {"english": self._title}}}}


def _gateway(client: _ScriptedTitleClient) -> AniListGateway:
    """A real gateway over the scripted wire client (its own cache leaf faked)."""

    return AniListGateway(cache_store=FakeCacheStore(), logger=make_logger(), client=client)


class _RecordingCacheStore(FakeCacheStore):
    """The in-memory store, also recording each `update_cache` payload so one merged write can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[CacheRecord] = []

    @override
    def update_cache(self, arr: Arr, al_id: int, cache_details: CacheRecord | None = None) -> None:
        self.updates.append(cache_details if cache_details is not None else {})
        super().update_cache(arr, al_id, cache_details)


def _unresolving() -> AniListGateway:
    """A gateway whose lookups resolve nothing, so a nameless cached entry stays nameless."""

    return _gateway(_ScriptedTitleClient(None))


class _RecordingReporter:
    """Records `log_cached_entry` calls, so the cross-arr param can be asserted on recorded state."""

    def __init__(self) -> None:
        self.calls: list[tuple[RunContext, Arr, int, EntryState]] = []

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        state: EntryState = EntryState.UNCHANGED,
    ) -> bool:
        self.calls.append((ctx, arr, al_id, state))
        return True


class TestCachedEntrySkip:
    """`cached_entry_skip` skips only a fresh cached entry, backfilling url/coverage once.

    It is bypassed by `ignore_seadex_update_times` or a dirty id, and the dirty set clears each run.
    """

    @staticmethod
    def _run(cache: FakeCacheStore) -> RunServices:
        # cached_entry_skip touches only cache_store + _config (real, default
        # ignore_seadex_update_times=False) + the reporter + the AniList gateway the
        # name backfill asks. ctx defaults to a SONARR RunContext (make_services).
        return make_services(cache_store=cache, _reporter=_RecordingReporter(), _anilist=_unresolving())

    def test_skips_when_cached_and_timestamp_matches(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        reporter = _RecordingReporter()
        run = make_services(cache_store=cache, _reporter=reporter, _anilist=_unresolving())
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True
        assert len(reporter.calls) == 1

    def test_does_not_skip_when_entry_absent(self) -> None:
        run = self._run(FakeCacheStore())
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_does_not_skip_when_timestamp_is_stale(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        # SeaDex entry now carries a newer updated_at -> stale -> re-process.
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2022, 6, 6)), lambda: "") is False

    def test_backfills_url_and_coverage_when_url_missing(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})  # legacy: no url yet
        run = self._run(cache)
        assert (
            run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1), url="sd-url"), lambda: "S01")
            is True
        )
        backfilled = cache.get_entry(Arr.SONARR, 7)
        assert backfilled is not None
        assert (backfilled.url, backfilled.coverage) == ("sd-url", "S01")

    def test_backfills_the_name_when_anilist_resolves_it_now(self) -> None:
        # A record written while AniList was unreachable carries no name. Once the
        # gateway resolves one, the skip stores it (nothing else re-processes a fresh entry).
        cache = _RecordingCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        cache.updates.clear()
        client = _ScriptedTitleClient("Resolved")
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _anilist=_gateway(client))

        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

        assert cache.updates == [{"name": "Resolved"}]
        assert client.query_calls == [7]

    def test_leaves_the_name_empty_when_anilist_still_has_none(self) -> None:
        cache = _RecordingCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        cache.updates.clear()
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=_gateway(_ScriptedTitleClient(None)),
        )

        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

        assert cache.updates == []
        entry = cache.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.name is None

    def test_leaves_a_stored_name_alone_without_a_lookup(self) -> None:
        cache = _RecordingCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"name": "Kept", "url": "u", "updated_at": datetime(2021, 1, 1)})
        cache.updates.clear()
        client = _ScriptedTitleClient("Resolved")
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), _anilist=_gateway(client))

        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

        assert cache.updates == []
        assert client.query_calls == []

    def test_url_and_name_backfills_merge_into_one_write(self) -> None:
        cache = _RecordingCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})  # legacy: no url, no name
        cache.updates.clear()
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _anilist=_gateway(_ScriptedTitleClient("Resolved")),
        )

        assert (
            run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1), url="sd-url"), lambda: "S01")
            is True
        )

        assert cache.updates == [{"url": "sd-url", "coverage": "S01", "name": "Resolved"}]

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
            _anilist=_unresolving(),
            _filter=_CtxBind(),
            _grab_pipeline=_CtxBind(),
        )
        run.mark_dirty([7])
        run.begin_run(run.ctx)
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True


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
            _anilist=_unresolving(),
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
        assert run.no_releases_skip(7, {"name": "Title"}) is False
        entry = cache.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is False


class _CtxBind:
    """A no-op ctx-bind collaborator for driving `begin_run` on a bare hub."""

    def begin_run(self, ctx: RunContext) -> None:
        del ctx


class TestCrossArrLookupHonorsParam:
    """`check_al_id_in_cache` / `log_cached_entry` read the `arr` parameter, never `self._ctx.arr`.

    The run-arr consolidation drops `arr` from methods that only ever act on the
    run's own arr, but these two stay parameterised because the Sonarr run's
    `ignore_movies_in_radarr` dedup calls them with `Arr.RADARR` to hit the
    Radarr cache while `ctx.arr` is SONARR. If either ever read `ctx.arr`
    instead of the param, that cross-arr check silently reads the wrong cache -
    a regression the all-SONARR tests above physically cannot catch.
    """

    def test_check_al_id_in_cache_honors_explicit_arr_over_ctx(self) -> None:
        cache = FakeCacheStore()
        # Present in the RADARR cache only. The run's ctx.arr is SONARR.
        cache.update_cache(Arr.RADARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = make_services(cache_store=cache, _ctx=RunContext(arr=Arr.SONARR))
        entry = make_entry_record(updated_at=datetime(2021, 1, 1))
        # The explicit RADARR param must select the Radarr cache (hit)...
        assert run.check_al_id_in_cache(Arr.RADARR, 7, entry) is True
        # ...and the run's own SONARR arr must miss (no Sonarr entry exists).
        assert run.check_al_id_in_cache(Arr.SONARR, 7, entry) is False

    def test_log_cached_entry_forwards_param_arr_not_ctx(self) -> None:
        reporter = _RecordingReporter()
        run = make_services(_reporter=reporter, _ctx=RunContext(arr=Arr.SONARR))
        run.log_cached_entry(Arr.RADARR, 7, state=EntryState.IN_RADARR)
        # The reporter delegate receives the explicit cross-arr value, not ctx.arr.
        assert reporter.calls == [(run._ctx, Arr.RADARR, 7, EntryState.IN_RADARR)]


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
        # The real four-step tail (log + cache write + throttle + False). Its body
        # previously had fake-only coverage: the seam tests script it, nothing
        # drove the real thing.
        cache = FakeCacheStore()
        reporter = _TailReporter()
        run = make_services(cache_store=cache, _reporter=reporter, sleep_time=0)

        assert run.no_releases_skip(7, {"name": "Title", "url": "u"}) is False

        persisted = cache.get_entry(Arr.SONARR, 7)
        assert persisted is not None
        assert (persisted.name, persisted.url) == ("Title", "u")
        assert len(reporter.no_releases_ctxs) == 1


class TestAlIdPrologue:
    """`al_id_prologue` reports a missing entry vs an outage distinctly, and resets skip flags/tallies on a hit."""

    def test_no_seadex_entry_reports_and_returns_none(self) -> None:
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(), _reporter=reporter)

        assert run.al_id_prologue(5, "Series") is None
        assert reporter.no_sd_entry_ids == [5]
        assert reporter.outage_skip_ids == []
        assert run._ctx.stats.checked == 1

    def test_outage_skip_reports_distinctly_and_returns_none(self) -> None:
        # A SeaDex-unreachable skip must never be reported as "no entry" - it
        # takes the outage reporter hook and its own tally.
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(outage=True), _reporter=reporter)

        assert run.al_id_prologue(5, "Series") is None
        assert reporter.outage_skip_ids == [5]
        assert reporter.no_sd_entry_ids == []
        assert run._ctx.stats.checked == 1

    def test_entry_found_resets_skip_flags_and_tallies(self) -> None:
        entry = make_entry_record()
        run = make_services(_seadex=FakeSeaDexSource({5: entry}), _reporter=_TailReporter())
        run._ctx.per_title.private_only_skipped = True  # stale flag from a previous title

        assert run.al_id_prologue(5, "Series") is entry
        assert run._ctx.per_title.private_only_skipped is False
        assert run._ctx.stats.checked == 1

    def test_seeds_the_arr_title_as_the_fallback(self) -> None:
        # The fresh per-title state carries the arr item's own title, so a later
        # title resolution can fall back to it when AniList has nothing.
        run = make_services(_seadex=FakeSeaDexSource({5: make_entry_record()}), _reporter=_TailReporter())

        run.al_id_prologue(5, "Series")

        assert run._ctx.per_title.arr_title == "Series"


class TestResolveTitle:
    """`resolve_title` shows AniList's title, else the arr's own, else the id form, and remembers it."""

    @staticmethod
    def _run(client: _ScriptedTitleClient, arr_title: str = "") -> RunServices:
        run = make_services(_anilist=_gateway(client))
        run._ctx.per_title.arr_title = arr_title
        return run

    def test_anilist_title_is_both_display_and_anilist(self) -> None:
        run = self._run(_ScriptedTitleClient("Resolved"), arr_title="Series")

        assert run.resolve_title(5) == EntryTitle(display="Resolved", anilist="Resolved")
        assert run._ctx.per_title.current_title == "Resolved"

    def test_falls_back_to_the_arr_title(self) -> None:
        run = self._run(_ScriptedTitleClient(None), arr_title="Series")

        assert run.resolve_title(5) == EntryTitle(display="Series", anilist=None)
        assert run._ctx.per_title.current_title == "Series"

    def test_falls_back_to_the_id_form_last(self) -> None:
        run = self._run(_ScriptedTitleClient(None))

        assert run.resolve_title(5) == EntryTitle(display="AniList #5", anilist=None)
        assert run._ctx.per_title.current_title == "AniList #5"


class TestNewCacheDetails:
    """`new_cache_details` seeds the record, carrying a name only when AniList resolved one."""

    def test_resolved_title_rides_as_the_name(self) -> None:
        entry = make_entry_record(updated_at=datetime(2021, 1, 1))
        run = make_services()

        details = run.new_cache_details(EntryTitle(display="Resolved", anilist="Resolved"), entry)

        assert details == {"name": "Resolved", "updated_at": entry.updated_at, "torrent_hashes": []}

    def test_fallback_title_is_never_stored_as_the_name(self) -> None:
        # A fallback label must not clobber a name a prior run stored (the merge
        # keeps the omitted field), nor persist an id form a later run would have to undo.
        entry = make_entry_record(updated_at=datetime(2021, 1, 1))
        run = make_services()

        details = run.new_cache_details(EntryTitle(display="Series", anilist=None), entry)

        assert details == {"updated_at": entry.updated_at, "torrent_hashes": []}
