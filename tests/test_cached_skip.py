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
from typing import Final

import pytest
from seadex import EntryRecord

from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import Title
from pearlarr.cache import CachedTitle, CachedTitleUpdate
from pearlarr.config import AppConfig, Arr
from pearlarr.output import Accent, StyledValue, TitleDetail, TitleHeader, TitleRow
from pearlarr.output.events import TitleState
from pearlarr.pending_import import TitleNames
from pearlarr.tvdb_orders import AIRED_ORDER

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    ScriptedAniListClient,
    anilist_body,
    make_anilist_gateway,
    make_config,
    make_entry_record,
    make_run_deps,
    make_services,
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


def _cached_row(*, named: bool = True, fallback_satisfied: bool = False) -> CachedTitleUpdate:
    """Title 7's row as a Sonarr check leaves it, or stored with no name when not `named`."""

    row = _checked()
    row["fallback_satisfied"] = fallback_satisfied
    if not named:
        del row["name"]
    return row


def _services(
    cache: FakeCacheStore, *, settings: AppConfig = _SETTINGS, anilist_name: str | None = None
) -> ArrRunServices:
    """One run's services over the cache, built as bootstrap builds them for each run.

    AniList knows title 7 by `anilist_name`, and nothing when it is None. qBittorrent is set up, so the run is no dry
    run and what it caches is what the next run reads.
    """

    deps = make_run_deps(config=settings, cache_store=cache, qbit=FakeQbit().client())
    if anilist_name is not None:
        deps.anilist.answers[7] = anilist_body(7, anilist_name)
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

    def test_a_full_row_is_left_alone_without_a_look_at_its_episodes_or_anilist(self) -> None:
        # Each look is a request: Sonarr's for the series, which the run's episode prefetch leaves out for a series
        # whose titles are all left alone, and AniList's once its answer for the title has aged out.
        def episodes() -> str:
            pytest.fail("looked up the episodes of a title whose row has its url")

        deps = make_run_deps(config=_SETTINGS, cache_store=_holding(_cached_row()), qbit=FakeQbit().client())

        assert ArrRunServices(deps, Arr.SONARR).is_unchanged(7, _ENTRY, episodes) is True
        assert deps.anilist.answers == {}, "asked AniList about a title whose row has its name"


class TestAnOlderRowIsFilledIn:
    """An unchanged title's row picks up the url, episodes and name an older run didn't store.

    Nothing else checks an unchanged title again, so this is the one place an older row can get them.
    """

    _COVERAGE_NOW = "S01 E01-E13"
    """Title 7's episodes in Sonarr now.

    One more than its row was cached with: a series numbered as aired gains episodes without its titles being sent on.
    """

    @pytest.mark.parametrize(
        ("stored", "anilist_name", "after"),
        [
            # The bare stamp is the whole row a run wrote before urls were stored.
            pytest.param(
                {"updated_at": _CACHED_ON},
                None,
                (None, _URL, _COVERAGE_NOW),
                id="a row from before urls were stored gets the url and episodes",
            ),
            pytest.param(
                _cached_row(named=False),
                "Show",
                ("Show", _URL, _COVERAGE),
                id="a row stored with no name gets AniList's name",
            ),
            pytest.param(
                {"updated_at": _CACHED_ON},
                "Show",
                ("Show", _URL, _COVERAGE_NOW),
                id="a row from before urls were stored gets AniList's name too",
            ),
            pytest.param(
                _cached_row(),
                "Renamed",
                ("Show", _URL, _COVERAGE),
                id="a stored name is kept",
            ),
            pytest.param(
                _cached_row(named=False),
                None,
                (None, _URL, _COVERAGE),
                id="no name is stored while AniList has none",
            ),
        ],
    )
    def test_an_unchanged_titles_row_gets_what_it_lacks(
        self, stored: CachedTitleUpdate, anilist_name: str | None, after: tuple[str | None, str, str]
    ) -> None:
        cache = _holding(stored)
        run = _services(cache, anilist_name=anilist_name)

        assert run.is_unchanged(7, _ENTRY, lambda: self._COVERAGE_NOW) is True

        row = cache.get_cached_title(Arr.SONARR, 7)
        assert row is not None
        assert (row.name, row.url, row.coverage) == after


class TestWhatTheNextRunChecks:
    """A title the gate sends on is checked again by every run until a check caches it.

    The gate clears the row's stamp before the check starts, so a check that fails, is held at the add cap, or stops
    for any reason before caching leaves the title to be checked again. Each run builds its services afresh over the
    same cache, as bootstrap does.
    """

    def test_a_title_left_alone_is_left_alone_next_run(self) -> None:
        cache = _holding(_cached_row())

        assert _services(cache).is_unchanged(7, _ENTRY, lambda: _COVERAGE) is True
        assert _services(cache).is_unchanged(7, _ENTRY, lambda: _COVERAGE) is True

    @pytest.mark.parametrize(
        "gate",
        [
            pytest.param(_Gate(files_changed=frozenset({7})), id="its files changed"),
            pytest.param(_Gate(last_scan=make_config(want_best=False)), id="the selection settings changed"),
        ],
    )
    def test_a_title_whose_check_caches_nothing_is_checked_again_next_run(self, gate: _Gate) -> None:
        cache = FakeCacheStore()
        assert _run_under(gate, cache).is_unchanged(7, _ENTRY, lambda: _COVERAGE) is False
        # The full scan stores the run's selection hash whatever became of each title.
        cache.store_selection_hash(Arr.SONARR, _SETTINGS.selection_hash())

        assert _services(cache).is_unchanged(7, _ENTRY, lambda: _COVERAGE) is False

    def test_a_title_whose_check_caches_it_is_left_alone_next_run(self) -> None:
        cache = _holding(_cached_row())
        run = _services(cache)
        run.mark_dirty([7])
        assert run.is_unchanged(7, _ENTRY, lambda: _COVERAGE) is False
        run.report_no_releases(7, _checked())

        assert _services(cache).is_unchanged(7, _ENTRY, lambda: _COVERAGE) is True


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


@pytest.mark.parametrize(
    ("seadex", "entry", "shown"),
    [
        pytest.param(FakeSeaDexSource({7: _ENTRY}), _ENTRY, [], id="an entry on SeaDex"),
        pytest.param(FakeSeaDexSource(), None, [TitleRow(TitleState.NO_ENTRY, "AniList #7")], id="no entry on SeaDex"),
        # An unreachable SeaDex must never read as "no entry".
        pytest.param(
            FakeSeaDexSource(unreachable=True),
            None,
            [TitleRow(TitleState.NOT_CHECKED, "AniList #7")],
            id="SeaDex unreachable",
        ),
    ],
)
def test_a_check_starts_by_finding_the_titles_entry(
    seadex: FakeSeaDexSource, entry: EntryRecord | None, shown: list[TitleRow]
) -> None:
    # A title with no entry to check stops there, shown with the reason.
    recording = install_recording_hub()
    run = ArrRunServices(make_run_deps(seadex=seadex), Arr.SONARR)

    assert (run.begin_check(7), recording.of_type(TitleRow)) == (entry, shown)


def test_the_run_counts_every_title_it_starts_on_even_one_seadex_couldnt_be_asked_about() -> None:
    # The summary gives that total, then a "not checked" line for the titles SeaDex couldn't be asked about. An
    # unreachable SeaDex still answers for the entries the run fetched before it went down.
    run = ArrRunServices(make_run_deps(seadex=FakeSeaDexSource({7: _ENTRY}, unreachable=True)), Arr.SONARR)

    run.begin_check(7)
    run.begin_check(8)

    assert run.state.stats.checked == 2


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
