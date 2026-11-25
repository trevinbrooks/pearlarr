# pyright: strict
"""Tests for the grab side (`GrabPipeline`): what a title's grab adds, tracks, caches, posts, and lists.

`make_grab_pipeline` builds the pipeline bare, so no qBittorrent login happens, and `FakeTorrents` stands in for
qBittorrent's add.
"""

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

import httpx
import pytest
import qbittorrentapi
from seadex import Tracker

from pearlarr import notify
from pearlarr.arr_run_state import ArrRunState, PerTitleState
from pearlarr.config import Arr
from pearlarr.discord import DiscordEmbed
from pearlarr.grab_pipeline import (
    GrabPipeline,
    GrabRequest,
    SkipReport,
)
from pearlarr.grab_placement import PendingImportUpdate
from pearlarr.notify import Notifier
from pearlarr.output import (
    CapReached,
    GrabFailed,
    NeedsActionCause,
    ReleaseSkipped,
    Severity,
    TitleDetail,
    severity_of,
)
from pearlarr.pending_import import OwnedAtGrab, PendingImport
from pearlarr.releases import (
    NO_ANOTHER_TITLES_COUNTS,
    AnotherTitlesCounts,
    EpisodeKey,
    PlacedEpisode,
    PrintedKey,
    Release,
    SeadexDict,
    SkipReason,
)
from pearlarr.stamps import now_stamp, stamp_of
from pearlarr.torrent_service import AddResult, GrabResult, TorrentAddError
from pearlarr.tracker_links import TrackerPageError
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    CLIENT_SENTINEL,
    PENDING_IMPORT_ANILIST_ID,
    AddStatus,
    FakeCacheStore,
    FakeSeaDexSource,
    FakeTorrents,
    ScriptedAniListClient,
    arr_files,
    claim_anilist_ids,
    grab_request,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release,
    make_services,
    one_release_dict,
    pending_import,
    pending_import_update,
    rg_group,
    sonarr_ep,
)
from .fakes import install_recording_hub


def _run_state(*, torrents_added: int = 0) -> ArrRunState:
    """A blocking Sonarr run checking the title named Show, `torrents_added` adds already counted toward the cap."""

    return ArrRunState(
        arr=Arr.SONARR,
        import_wait_mode=ImportWaitMode.BLOCKING,
        torrents_added=torrents_added,
        per_title=PerTitleState(title_name="Show"),
    )


def _pipeline(
    *,
    torrents: FakeTorrents,
    qbit: object = CLIENT_SENTINEL,
    state: ArrRunState | None = None,
    **overrides: object,
) -> GrabPipeline:
    """A bare `GrabPipeline` wired for the add path (a blocking run, not a dry run), on `state` when one is given.

    `overrides` go to `make_grab_pipeline`: config settings, or a part of the pipeline to swap in.
    """

    return make_grab_pipeline(_torrents=torrents, qbit=qbit, _state=state or _run_state(), **overrides)


def _grab_request(seadex_dict: SeadexDict, *, input_missing: bool = False) -> GrabRequest:
    """Title 42's request over `seadex_dict`, its one group among the failed reads when `input_missing`."""

    return grab_request(
        anilist_id=42,
        entry=make_entry_record(url="https://seadex.example/42"),
        seadex_dict=seadex_dict,
        update={"updated_at": "2026-01-01 00:00:00"},
        input_unread_groups=("RG",) if input_missing else (),
    )


def _pending_import_rows(pipeline: GrabPipeline) -> Mapping[str, dict[str, Any]]:
    """The pipeline's durable per-arr pending import store, keyed by infohash (what the arr run reads back)."""

    return pipeline.cache_store.get_pending_import_rows(Arr.SONARR)


def _stored(pipeline: GrabPipeline, infohash: str) -> PendingImport:
    """The store's pending import on the torrent, loaded (its stamps and claims need no guard row)."""

    raw = pipeline.cache_store.get_pending_import_row(Arr.SONARR, infohash)
    assert raw is not None
    return PendingImport.from_json(raw, guards={})


def _guards(pipeline: GrabPipeline) -> Mapping[int, OwnedAtGrab]:
    """The pipeline's durable per-title guard rows (what the reads load from)."""

    return pipeline.cache_store.get_guards(Arr.SONARR)


def _nyaa_release(*, url: str, infohash: str) -> Release:
    """A Nyaa release marked to grab that clears every add-path filter."""

    item = make_release(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return item


def _anidex_release(*, url: str, infohash: str) -> Release:
    """A release marked to grab on AniDex: public (clears the private-only gate), in the default tracker set.

    It has no parser, so the add loop skips it.
    """

    item = make_release(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.ANIDEX
    return item


def _release(n: int, skipped: SkipReason | None = None) -> Release:
    """Release `n` (hash `h<n>`) marked to grab: skipped for `skipped` by its tracker or its placement, else grabbable."""

    url, infohash = f"https://tracker.example/{n}", f"h{n}"
    if skipped is SkipReason.UNSUPPORTED_TRACKER:
        return _anidex_release(url=url, infohash=infohash)
    item = _nyaa_release(url=url, infohash=infohash)
    if skipped is SkipReason.PRIVATE_ONLY:
        item.tracker, item.is_public = Tracker.ANIMEBYTES, False
    else:
        item.skip_reason = skipped
    return item


# The add loop skips for these itself, by the release's tracker.
_ADD_TIME_SKIPS = {SkipReason.PRIVATE_ONLY, SkipReason.UNSUPPORTED_TRACKER, SkipReason.TRACKER_NOT_SELECTED}


# Highest first: the one needs-action row a title gets when two of these meet on it. None is a failed grab.
_ROW_PRECEDENCE: list[tuple[SkipReason | None, NeedsActionCause]] = [
    (SkipReason.PRIVATE_ONLY, NeedsActionCause.PRIVATE_ONLY),
    (SkipReason.UNSUPPORTED_TRACKER, NeedsActionCause.UNSUPPORTED_TRACKER),
    (SkipReason.MISNUMBERED, NeedsActionCause.MISNUMBERED),
    (SkipReason.EPISODE_TITLES_CONTRADICT, NeedsActionCause.EPISODE_TITLES_CONTRADICT),
    (None, NeedsActionCause.GRAB_FAILED),
    (SkipReason.PLACED_ELSEWHERE, NeedsActionCause.PLACED_ELSEWHERE),
    (SkipReason.PLACEMENT_INPUT_UNREAD, NeedsActionCause.PLACEMENT_INPUT_UNREAD),
]


class TestAGrab:
    """A title's grab: what it lists, caches, and counts."""

    def test_the_grabbed_row_names_the_placed_episodes_by_the_numbers_they_print_by(self) -> None:
        # Sonarr lists the aired S02E01 as S01E04.
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")}), state=state)
        seadex = one_release_dict(srg="NAN0", infohash="h1")
        (release,) = seadex["NAN0"].releases.values()
        release.episodes = [PlacedEpisode(2, 1, 10, printed=PrintedKey(EpisodeKey(1, 4)))]

        pipeline.grab_releases(
            grab_request(seadex_dict=seadex, pending_import_updates={"h1": pending_import_update("h1")})
        )

        assert [row.coverage for row in state.stats.grabbed] == ["S01 E04"]

    def test_a_grab_with_no_fallback_in_it_clears_the_titles_stored_fallback_marker(self) -> None:
        # The cache write merges columns, so a marker an earlier fallback grab left stays unless every write sets it.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}))
        pipeline.cache_store.update_cached_title(Arr.SONARR, 42, {"fallback_satisfied": True})

        pipeline.grab_and_cache(_grab_request(one_release_dict(srg="RG", infohash="h1")))

        cached = pipeline.cache_store.get_cached_title(Arr.SONARR, 42)
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_two_up_to_date_titles_both_counted(self) -> None:
        # Two titles with nothing to grab: the summary counts both.
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state)

        for anilist_id in (1, 2):
            pipeline.grab_and_cache(
                grab_request(anilist_id=anilist_id, entry=make_entry_record(url=f"https://seadex.example/{anilist_id}"))
            )

        assert state.stats.up_to_date == 2


class TestGrabReleaseRegistersPending:
    """A grabbed release's pending import is stored from its update, and a fresh add is a grab that counts toward the cap."""

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)
        facts = OwnedAtGrab(entry_groups=("NAN0",), stale_groups=("OldPick",))
        updates = {"h1": pending_import_update("h1", guards=facts)}

        n_added, _ = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert set(_pending_import_rows(pipeline)) == {"h1"}
        assert _guards(pipeline) == {PENDING_IMPORT_ANILIST_ID: facts}
        assert [p.infohash for p in state.grabbed_this_run.values()] == ["h1"]
        assert n_added == 1
        assert state.torrents_added == 1
        assert len(state.stats.grabbed) == 1

    def test_a_second_pending_import_update_carrying_the_first_pending_import_joins_its_claim(self) -> None:
        # Two titles share one torrent in one run: A's add tracks the pending import fresh, B's dedups to
        # ALREADY_IN_QBIT with an update carrying A's pending import (the store read). B's claim joins it, the
        # run-list copy follows, and a this-run grab is never a reacquire.
        store = FakeCacheStore()
        state = _run_state()
        first = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}), state=state, cache_store=store)
        first.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_import_updates={"h1": pending_import_update("h1", anilist_id=11, title="Cour 1")},
            )
        )
        second = _pipeline(
            torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), state=state, cache_store=store
        )
        update = pending_import_update("h1", anilist_id=22, title="Cour 2", stored=_stored(first, "h1"))

        second.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={"h1": update})
        )

        assert claim_anilist_ids(_stored(second, "h1")) == (11, 22)
        assert claim_anilist_ids(state.grabbed_this_run["h1"]) == (11, 22)
        assert state.reacquired_keys == set()
        assert _guards(second).keys() == {11, 22}

    def test_dry_run_does_not_register_but_returns_outcome(self) -> None:
        # No client -> dry run: nothing persisted, but the outcome still surfaces.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "x")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, qbit=None, state=state)
        updates = {"h1": pending_import_update("h1")}

        _, results = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert _pending_import_rows(pipeline) == {}
        assert _guards(pipeline) == {}
        assert state.grabbed_this_run == {}
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT]

    def test_radarr_registration_writes_no_guard_row(self) -> None:
        # Guard evidence is Sonarr-only enforcement: a Radarr state driven through
        # the same path persists its pending import but never a guard row.
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Movie")})
        pipeline = make_grab_pipeline(
            _torrents=torrents,
            _state=ArrRunState(arr=Arr.RADARR, import_wait_mode=ImportWaitMode.BLOCKING),
        )
        updates = {"h1": pending_import_update("h1", series_id=0)}

        pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates)
        )

        assert set(pipeline.cache_store.get_pending_import_rows(Arr.RADARR)) == {"h1"}
        assert pipeline.cache_store.get_guards(Arr.RADARR) == {}
        assert pipeline.cache_store.get_guards(Arr.SONARR) == {}


class TestReacquireRegistration:
    """A torrent already in qBittorrent joins the stored pending import its update carries, or starts one.

    A new one starts at qBittorrent's add time. A fresh add of a torrent with a stored pending import restarts it now.
    """

    _STORED_AT = "2026-01-01 00:00:00"

    def _reacquire(self, added_on: datetime | None) -> FakeTorrents:
        return FakeTorrents({"h1": AddResult(AddStatus.ALREADY_IN_QBIT, "Show", added_on)})

    def _resident(self, pipeline: GrabPipeline) -> PendingImport:
        """A carried-over pending import put in the store, stamped well before this run."""

        resident = pending_import(infohash="h1", added_at=self._STORED_AT)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, resident.infohash, resident.to_json())
        return resident

    def _add(self, pipeline: GrabPipeline, update: PendingImportUpdate) -> list[GrabResult]:
        _, results = pipeline.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={"h1": update}
            ),
        )
        return results

    def test_a_joining_title_keeps_the_stored_start(self) -> None:
        # A second title's update carrying the stored pending import joins it. The pending import keeps its own
        # `added_at` whatever qBittorrent's add time says, and a carried-over pending import is a reacquire (never a
        # run-list insert).
        state = _run_state()
        pipeline = _pipeline(torrents=self._reacquire(datetime.now() - timedelta(days=100)), state=state)
        resident = self._resident(pipeline)

        self._add(pipeline, pending_import_update("h1", anilist_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert pending.added_at == self._STORED_AT
        assert [c.anilist_id for c in pending.claims] == [PENDING_IMPORT_ANILIST_ID, 22]
        assert state.reacquired_keys == {"h1"}
        assert state.grabbed_this_run == {}

    def test_resident_reacquire_refreshes_the_guard_row(self) -> None:
        # The title's guard evidence follows the newest plan, never a frozen copy:
        # a claim joining a carried-over pending import re-puts the row the trust read loads.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        facts = OwnedAtGrab(entry_groups=("NAN0",))

        self._add(pipeline, pending_import_update("h1", guards=facts, stored=resident))

        assert _guards(pipeline) == {PENDING_IMPORT_ANILIST_ID: facts}

    def test_a_torrent_long_in_qbittorrent_is_tracked_at_its_add_time(self) -> None:
        # No stored pending import: the new one starts at qBittorrent's add time, however long ago that was.
        added_on = datetime.now() - timedelta(days=100)
        state = _run_state()
        pipeline = _pipeline(torrents=self._reacquire(added_on), state=state)

        results = self._add(pipeline, pending_import_update("h1"))

        assert _stored(pipeline, "h1").added_at == stamp_of(added_on)
        assert state.reacquired_keys == {"h1"}
        assert state.grabbed_this_run == {}
        assert _guards(pipeline).keys() == {PENDING_IMPORT_ANILIST_ID}
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT]

    def test_a_new_pending_import_with_no_add_time_starts_now(self) -> None:
        pipeline = _pipeline(torrents=self._reacquire(None))
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1"))

        assert before <= _stored(pipeline, "h1").added_at <= now_stamp()

    def test_a_marked_release_is_tracked_under_its_cross_seed(self) -> None:
        # The torrent that already has the release's files is in qBittorrent under another hash than SeaDex's.
        pipeline = _pipeline(torrents=self._reacquire(None))
        seadex_dict = one_release_dict(srg="NAN0", infohash="h1")
        (release,) = seadex_dict["NAN0"].releases.values()
        release.cross_seed_hash = "c1"

        pipeline.grab_releases(
            grab_request(seadex_dict=seadex_dict, pending_import_updates={"c1": pending_import_update("c1")})
        )

        assert _pending_import_rows(pipeline).keys() == {"c1"}

    def test_fresh_add_of_a_stored_pending_import_restamps_the_start(self) -> None:
        # A re-add of a torrent whose pending import survived (qBittorrent lost it): the
        # joined pending import enters the run list restarted at now, the start a new
        # download's rescan proof counts from.
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}), state=state)
        resident = self._resident(pipeline)
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1", anilist_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert [c.anilist_id for c in pending.claims] == [PENDING_IMPORT_ANILIST_ID, 22]
        assert before <= pending.added_at <= now_stamp()
        assert state.reacquired_keys == set()
        assert list(state.grabbed_this_run) == ["h1"]


class TestTheRunCap:
    """A title's grabbable releases past the run's cap of adds are held, and the title says so and stays uncached.

    The title whose adds reach the cap announces it, once. A run with no cap in force holds none and announces none.
    """

    def test_cap_holds_the_releases_past_it(self) -> None:
        # Three releases to grab under a cap of 2: the first two reach qBittorrent and count toward the cap, and
        # the third is held, never asked of qBittorrent.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        u3 = _nyaa_release(url="https://nyaa.si/view/3", infohash="h3")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2, u3.url: u3})}
        torrents = FakeTorrents(
            {
                "h1": (AddStatus.ADDED, "one"),
                "h2": (AddStatus.ADDED, "two"),
                "h3": (AddStatus.ADDED, "three"),
            }
        )
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=2)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2"]
        assert n_added == 2
        assert state.torrents_added == 2
        assert [r.outcome for r in results] == [AddStatus.ADDED, AddStatus.ADDED]
        assert state.stats.held_by_cap == 1

    def test_a_title_past_the_cap_is_held_and_tallied_once(self) -> None:
        # Two grabbable releases past the cap: one tally, no qBittorrent call.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one"), "h2": (AddStatus.ADDED, "two")})
        state = _run_state(torrents_added=1)
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == []
        assert (n_added, results) == (0, [])
        assert state.stats.held_by_cap == 1

    def test_a_joining_title_held_by_the_cap_saves_its_claim(self) -> None:
        # A torrent already in qBittorrent under a stored pending import stays tracked past the cap: the title's claim
        # joins the pending import without a qBittorrent call, and nothing joins the reacquired keys.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one")})
        state = _run_state(torrents_added=1)
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)
        resident = pending_import(infohash="h1", anilist_id=11)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", anilist_id=22, stored=resident)}

        pipeline.grab_releases(grab_request(seadex_dict=seadex_dict, pending_import_updates=updates))

        assert torrents.calls == []
        assert claim_anilist_ids(_stored(pipeline, "h1")) == (11, 22)
        assert state.reacquired_keys == set()
        assert state.grabbed_this_run == {}

    def test_a_torrent_already_in_qbittorrent_is_no_grab_and_the_release_after_it_still_adds(self) -> None:
        # Under a cap of 1, a torrent already there that counted toward it would hold the release after it.
        old = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        new = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "old"), "h2": (AddStatus.ADDED, "new")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)

        n_added, results = pipeline.grab_releases(
            grab_request(seadex_dict={"RG": rg_group({old.url: old, new.url: new})})
        )

        assert torrents.calls == ["h1", "h2"]
        assert (n_added, state.torrents_added) == (1, 1)
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT, AddStatus.ADDED]
        assert [row.name for row in state.stats.grabbed] == ["new"]
        # The request carries no update for either torrent, so neither is tracked.
        assert _pending_import_rows(pipeline) == {}

    @pytest.mark.parametrize(
        ("hashes", "already_added", "added", "held"),
        [
            pytest.param(["h1"], 0, ["h1"], False, id="reaches-the-cap"),
            pytest.param(["h1"], 1, [], True, id="past-the-cap"),
            pytest.param(["h1", "h2"], 0, ["h1"], True, id="crosses-the-cap-mid-title"),
        ],
    )
    def test_only_a_release_past_the_cap_holds_its_title_and_the_title_reaching_the_cap_announces_it(
        self, hashes: list[str], already_added: int, added: list[str], held: bool
    ) -> None:
        recording = install_recording_hub()
        releases = [_nyaa_release(url=f"https://nyaa.si/view/{infohash}", infohash=infohash) for infohash in hashes]
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one"), "h2": (AddStatus.ADDED, "two")})
        state = _run_state(torrents_added=already_added)
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)
        request = _grab_request({"RG": rg_group({release.url: release for release in releases})})
        updates = {infohash: pending_import_update(infohash) for infohash in hashes}

        pipeline.grab_and_cache(replace(request, pending_import_updates=updates))

        statuses = [d.value.text for d in recording.of_type(TitleDetail) if d.label == "status"]
        assert statuses == (["held by the run cap; grabbed next run"] if held else [])
        assert torrents.calls == added
        # Under a cap of 1, a title that adds anything is the one whose add reaches the cap.
        assert [e.cap for e in recording.of_type(CapReached)] == ([1] if added else [])
        # A held release that is new to qBittorrent is not tracked yet.
        assert _pending_import_rows(pipeline).keys() == set(added)
        assert (pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None) == held
        assert (state.stats.held_by_cap, state.stats.needs_action) == (int(held), [])

    @pytest.mark.parametrize(
        ("cap", "qbit"),
        [pytest.param(0, CLIENT_SENTINEL, id="cap-of-zero"), pytest.param(1, None, id="dry-run")],
    )
    def test_a_run_with_no_cap_in_force_adds_every_release_and_announces_no_cap(self, cap: int, qbit: object) -> None:
        recording = install_recording_hub()
        one = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        two = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one"), "h2": (AddStatus.ADDED, "two")})
        pipeline = _pipeline(torrents=torrents, qbit=qbit, max_torrents_to_add=cap)

        pipeline.grab_and_cache(_grab_request({"RG": rg_group({one.url: one, two.url: two})}))

        assert torrents.calls == ["h1", "h2"]
        assert recording.of_type(CapReached) == []

    @pytest.mark.parametrize(
        "skipped", [SkipReason.PRIVATE_ONLY, SkipReason.MISNUMBERED], ids=["private", "misnumbered"]
    )
    def test_a_release_screened_out_past_the_cap_posts_its_skip_and_holds_nothing(self, skipped: SkipReason) -> None:
        recording = install_recording_hub()
        release = _release(1, skipped)
        state = _run_state(torrents_added=1)
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state, max_torrents_to_add=1)

        pipeline.grab_releases(grab_request(seadex_dict={"RG": rg_group({release.url: release})}))

        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [skipped]
        assert state.stats.held_by_cap == 0


class TestTrackerSkips:
    """A title whose releases are all on a private tracker or one with no parser grabs nothing and stays uncached.

    It gets one needs-action row saying why.
    """

    @pytest.mark.parametrize(
        ("skipped", "config", "reason", "cause"),
        [
            pytest.param(
                [SkipReason.UNSUPPORTED_TRACKER],
                {},
                "tracker not yet supported; grab manually",
                NeedsActionCause.UNSUPPORTED_TRACKER,
                id="no-parser",
            ),
            # The two reasons are either/or: private-only wins.
            pytest.param(
                [SkipReason.PRIVATE_ONLY, SkipReason.UNSUPPORTED_TRACKER],
                {"private_releases": "warn"},
                "private-only release; private releases not supported",
                NeedsActionCause.PRIVATE_ONLY,
                id="private-beside-no-parser",
            ),
            # Fallback mode and still nothing grabbable: no fallback covered the title's files. Its own cause keeps
            # the summary tip from suggesting the fallback that's already on.
            pytest.param(
                [SkipReason.PRIVATE_ONLY],
                {"private_releases": "fallback"},
                "private-only release; no fallback covers these files",
                NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
                id="private-in-fallback-mode",
            ),
            # Interactive and fallback: the private release was picked by hand, and the cause stays the one whose
            # tip never suggests the fallback.
            pytest.param(
                [SkipReason.PRIVATE_ONLY],
                {"private_releases": "fallback", "interactive": True},
                "hand-picked private release; private releases not supported",
                NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
                id="hand-picked-private",
            ),
        ],
    )
    def test_a_title_with_nothing_grabbable_is_left_uncached_under_one_row_saying_why(
        self, skipped: list[SkipReason], config: dict[str, object], reason: str, cause: NeedsActionCause
    ) -> None:
        recording = install_recording_hub()
        releases = [_release(n, skip) for n, skip in enumerate(skipped, start=1)]
        torrents = FakeTorrents({})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state, **config)

        pipeline.grab_and_cache(_grab_request({"NAN0": rg_group({release.url: release for release in releases})}))

        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == skipped
        assert torrents.calls == []
        assert [(row.group, row.reason, row.cause) for row in state.stats.needs_action] == [("NAN0", reason, cause)]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None


@pytest.mark.parametrize(
    ("beside", "config", "cause"),
    [
        pytest.param(SkipReason.UNSUPPORTED_TRACKER, {}, None, id="no-parser"),
        pytest.param(
            SkipReason.PRIVATE_ONLY,
            {"private_releases": "fallback", "interactive": True},
            None,
            id="hand-picked-private",
        ),
        pytest.param(SkipReason.MISNUMBERED, {}, NeedsActionCause.MISNUMBERED, id="misnumbered"),
        pytest.param(None, {}, NeedsActionCause.GRAB_FAILED, id="failed-grab"),
    ],
)
def test_a_grab_caches_its_title_past_a_tracker_skip_and_never_past_a_placement_skip_or_a_failed_grab(
    beside: SkipReason | None, config: dict[str, object], cause: NeedsActionCause | None
) -> None:
    # The one add reaches the cap of 1, which changes none of it.
    first, second = _release(1, beside), _release(2)
    torrents = FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}, raises={"h1": httpx.ConnectError("nyaa down")})
    state = _run_state()
    pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1, **config)

    pipeline.grab_and_cache(_grab_request({"RG": rg_group({first.url: first, second.url: second})}))

    assert state.torrents_added == 1
    assert [row.cause for row in state.stats.needs_action] == ([] if cause is None else [cause])
    assert (pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None) == (cause is not None)


class TestPlacementSkips:
    """A release the placement skipped is never grabbed: the title stays uncached, and the summary says why.

    One skipped over episodes Sonarr has no row for is never grabbed either, and its title is cached.
    """

    @pytest.mark.parametrize(
        ("skip_reason", "counts", "expected"),
        [
            pytest.param(
                SkipReason.MISNUMBERED,
                NO_ANOTHER_TITLES_COUNTS,
                SkipReport(
                    NeedsActionCause.MISNUMBERED,
                    "numbering does not match its SeaDex listing; import it by hand",
                ),
                id="misnumbered",
            ),
            pytest.param(
                SkipReason.EPISODE_TITLES_CONTRADICT,
                NO_ANOTHER_TITLES_COUNTS,
                SkipReport(
                    NeedsActionCause.EPISODE_TITLES_CONTRADICT,
                    "episode titles contradict its numbers; import it by hand",
                ),
                id="titles-refute",
            ),
            pytest.param(
                SkipReason.PLACED_ELSEWHERE,
                AnotherTitlesCounts(another_titles=12, unplaced=6),
                SkipReport(
                    NeedsActionCause.PLACED_ELSEWHERE,
                    "its files place under another title SeaDex lists it with; any it could not place need a hand "
                    "import",
                ),
                id="another-titles-files",
            ),
        ],
    )
    def test_a_skipped_release_is_never_added_and_its_skip_and_row_say_why(
        self, skip_reason: SkipReason, counts: AnotherTitlesCounts, expected: SkipReport
    ) -> None:
        recording = install_recording_hub()
        release = _release(1, skip_reason)
        release.another_titles_counts = counts
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request({"RG": rg_group({release.url: release})}))

        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [(e.reason, e.counts) for e in recording.of_type(ReleaseSkipped)] == [(skip_reason, counts)]
        (row,) = state.stats.needs_action
        assert (row.cause, row.group, row.reason) == (expected.cause, "RG", expected.reason)

    def test_a_release_on_episodes_sonarr_has_no_row_for_is_planned_but_never_added_and_its_title_is_cached(
        self,
    ) -> None:
        # The planner marks the release to grab by its group and sizes: none of its files placed, and the title's
        # one episode in Sonarr has no file.
        recording = install_recording_hub()
        nyaa = make_release(url="https://nyaa.si/view/1", infohash="h1", size=[100, 200])
        nyaa.tracker = Tracker.NYAA
        nyaa.skip_reason = SkipReason.AIRED_ONLY
        in_sonarr = [sonarr_ep(1, 3, episode_file_id=0)]
        plan = make_planner().plan({"RG": rg_group({nyaa.url: nyaa}, episodes=[])}, arr_files(in_sonarr), in_sonarr)
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request(plan.seadex_dict))

        assert nyaa.to_grab is True
        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is not None
        assert [(e.reason, e.severity) for e in recording.of_type(ReleaseSkipped)] == [
            (SkipReason.AIRED_ONLY, Severity.INFO)
        ]
        assert state.stats.needs_action == []

    @pytest.mark.parametrize(
        "skip_reason",
        [pytest.param(reason, id=reason.name.lower()) for reason in SkipReason if reason not in _ADD_TIME_SKIPS],
    )
    def test_a_release_skipped_for_any_placement_reason_is_never_added_and_its_title_is_listed_or_cached(
        self, skip_reason: SkipReason
    ) -> None:
        # A reason with no row of its own that left its title uncached would have it checked every run in silence.
        recording = install_recording_hub()
        release = _release(1, skip_reason)
        torrents = FakeTorrents({})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request({"RG": rg_group({release.url: release})}))

        cause = dict(_ROW_PRECEDENCE).get(skip_reason)
        assert torrents.calls == []
        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [skip_reason]
        assert [row.cause for row in state.stats.needs_action] == ([] if cause is None else [cause])
        assert (pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None) == (cause is not None)

    def test_a_skipped_release_of_a_resident_torrent_never_joins_its_pending_import(self) -> None:
        # Another title's pending import has the torrent.
        resident = pending_import(infohash="h1")
        release = _release(1, SkipReason.EPISODE_TITLES_CONTRADICT)
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}))
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", anilist_id=22, stored=resident)}

        pipeline.grab_releases(
            grab_request(seadex_dict={"RG": rg_group({release.url: release})}, pending_import_updates=updates)
        )

        assert claim_anilist_ids(_stored(pipeline, "h1")) == (PENDING_IMPORT_ANILIST_ID,)

    def test_a_release_not_to_grab_with_a_skip_reason_is_not_skipped_and_the_title_caches(self) -> None:
        # The arr has the torrent's files now (the hand import is done), and another release of the title still grabs.
        recording = install_recording_hub()
        done, other = _release(1, SkipReason.MISNUMBERED), _release(2)
        done.to_grab = False
        pipeline = _pipeline(torrents=FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}))

        pipeline.grab_and_cache(_grab_request({"RG": rg_group({done.url: done, other.url: other})}))

        assert recording.of_type(ReleaseSkipped) == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is not None


class TestPlacementInputUnread:
    """A failed Sonarr read under a grab-time placement leaves the title uncached, with a retry row in the summary."""

    def test_a_grabbed_title_stays_uncached_with_a_retry_row(self) -> None:
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), state=state)

        pipeline.grab_and_cache(_grab_request(one_release_dict(srg="RG", infohash="h1"), input_missing=True))

        assert [row.name for row in state.stats.grabbed] == ["Show-RG"]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]
        assert rows[0].reason == "a read the placement needs failed; will retry next run"
        assert rows[0].group == "RG"

    def test_a_title_with_nothing_to_grab_is_neither_up_to_date_nor_cached(self) -> None:
        # The placement may have left a sequence waiting, so the coverage decision was coarse: no "already have it".
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=False)})}
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state)

        pipeline.grab_and_cache(_grab_request(seadex_dict, input_missing=True))

        assert state.stats.up_to_date == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [r.cause for r in state.stats.needs_action] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]


def test_a_title_read_without_its_metadata_stays_uncached_with_nothing_to_do() -> None:
    # AniList or SkyHook was unreachable before the first fetch: the title re-checks next run, and no row asks for a
    # hand.
    seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=False)})}
    state = _run_state()
    pipeline = _pipeline(torrents=FakeTorrents({}), state=state)

    pipeline.grab_and_cache(replace(_grab_request(seadex_dict), metadata_unread=True))

    assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
    assert state.stats.needs_action == []


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent unreachable or erroring) is contained at the add.

    One warning, no traceback: the release loop moves on, the title stays uncached (retried next run), and the summary
    carries a GRAB_FAILED needs-action row instead of the title silently vanishing.
    """

    @pytest.mark.parametrize(
        "error",
        [
            TrackerPageError("Could not find the torrent title on https://nyaa.si/view/1"),
            TorrentAddError("qBittorrent rejected the torrent"),
            httpx.ConnectError("tracker down"),
            qbittorrentapi.APIConnectionError("qbit died mid-run"),
        ],
        ids=["parse", "add", "tracker", "qbit"],
    )
    def test_failure_is_one_clean_warning_no_traceback(self, error: Exception) -> None:
        # Each kind of error a tracker or qBittorrent raises is caught at the add, and posted as one warning naming
        # the release.
        torrents = FakeTorrents({}, raises={"h1": error})
        pipeline = _pipeline(torrents=torrents)
        recording = install_recording_hub()

        n_added, results = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1")),
        )

        assert n_added == 0
        assert results == []
        (failed,) = recording.of_type(GrabFailed)
        assert failed.group == "NAN0"
        assert failed.url == "https://nyaa.si/view/1"
        assert failed.error == str(error)
        assert severity_of(failed) is Severity.WARNING

    def test_failed_release_does_not_drop_the_next_one(self) -> None:
        # Containment is per release: the sibling release after the failure still grabs.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddStatus.ADDED, "Show-RG")},
            raises={"hbad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["hbad", "hgood"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddStatus.ADDED]

    def test_failed_only_title_stays_uncached_with_a_retry_row(self) -> None:
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request(one_release_dict(srg="RG", infohash="h1")))

        assert state.stats.grabbed == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.GRAB_FAILED]
        assert rows[0].reason == "grab failed; will retry next run"
        assert rows[0].group == "RG"

    def test_a_failed_grab_on_one_title_never_keeps_the_next_title_uncached(self) -> None:
        # Two titles in one run, each through the real per-title head: the first title's add fails, the second's
        # goes through.
        first, second = _release(1), _release(2)
        torrents = FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}, raises={"h1": httpx.ConnectError("nyaa down")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)
        entries = {1: make_entry_record(anilist_id=1), 2: make_entry_record(anilist_id=2)}
        services = make_services(_seadex=FakeSeaDexSource(entries), _state=state, _grab_pipeline=pipeline)

        for anilist_id, release in ((1, first), (2, second)):
            entry = services.begin_check(anilist_id)
            assert entry is not None
            services.grab_and_cache(
                grab_request(
                    anilist_id=anilist_id,
                    entry=entry,
                    seadex_dict={"RG": rg_group({release.url: release})},
                    update={"updated_at": "2026-01-01 00:00:00"},
                )
            )

        assert [row.cause for row in state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 1) is None
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 2) is not None


@pytest.mark.parametrize(("wins", "loses"), list(combinations(_ROW_PRECEDENCE, 2)), ids=lambda row: row[1].value)
def test_two_reasons_on_one_title_list_it_once_under_the_one_that_outranks_the_other(
    wins: tuple[SkipReason | None, NeedsActionCause], loses: tuple[SkipReason | None, NeedsActionCause]
) -> None:
    # The losing release comes first, so the order they are met in decides nothing.
    first, second = _release(1, loses[0]), _release(2, wins[0])
    down = httpx.ConnectError("nyaa down")
    state = _run_state()
    pipeline = _pipeline(torrents=FakeTorrents({}, raises={"h1": down, "h2": down}), state=state)

    pipeline.grab_and_cache(_grab_request({"A": rg_group({first.url: first}), "B": rg_group({second.url: second})}))

    assert [row.cause for row in state.stats.needs_action] == [wins[1]]


class TestGrabPushesNotice:
    """A title's grab posts one Discord notice built from its request and what the adds did.

    Never in a dry run, and never when nothing was added.
    """

    @staticmethod
    def _grab(
        monkeypatch: pytest.MonkeyPatch,
        torrents: FakeTorrents,
        seadex_dict: SeadexDict | None = None,
        *,
        qbit: object = CLIENT_SENTINEL,
    ) -> list[DiscordEmbed]:
        """The embeds posted for title 42's grab of `seadex_dict` (group PMR's one release h1 when not given)."""

        embeds: list[DiscordEmbed] = []

        def record(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
            del url, client
            embeds.append(embed)

        monkeypatch.setattr(notify, "discord_push", record)
        pipeline = _pipeline(
            torrents=torrents,
            qbit=qbit,
            _notifier=Notifier(
                discord_url="https://discord.example",
                webhook_url=None,
                web=httpx.Client(),
            ),
            # AniList knows the title, so its cover and banner frame the notice.
            anilist_client=ScriptedAniListClient("Show Title"),
        )
        pipeline.grab_and_cache(
            GrabRequest(
                anilist_id=42,
                # The library item's name must not prefix the AniList one, or the byline
                # this class asserts on would (correctly) dedupe away.
                arr_title="The Show",
                title_name="Show Title",
                entry=make_entry_record(url="https://releases.moe/42", notes="the why"),
                seadex_dict=seadex_dict or one_release_dict(srg="PMR", infohash="h1"),
                update={},
                replaced_groups=("OldGroup",),
                coverage="S01 E01-E12",
            ),
        )
        return embeds

    def test_added_pushes_the_resolved_notice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        [embed] = self._grab(monkeypatch, FakeTorrents({"h1": (AddStatus.ADDED, "Show-PMR")}))

        # The request's entry/coverage and AniList's art all reached the embed.
        assert embed.title == "Show Title"
        assert embed.url == "https://releases.moe/42"
        assert embed.thumb_url == "https://img/large"
        assert embed.image_url == "https://img/banner"
        # A single-group grab hoists its release into the description. The
        # subtitle/notes stack trails as the nameless (header-free) field.
        assert embed.description == "**Grabbed · `PMR`**\n[Nyaa](https://nyaa.si/view/1)"
        assert [f.name for f in embed.fields] == ["Episodes", "Replacing", ""]
        assert embed.fields[-1].value == "-# The Show\n> the why"

    def test_a_group_whose_add_failed_is_named_failed_beside_the_grabbed_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nyaa was down for group Bad's release. Its block must not read as skipped, which would say it was never tried.
        grabbed = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        failed = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-PMR")}, raises={"h2": httpx.ConnectError("nyaa down")})

        [embed] = self._grab(
            monkeypatch, torrents, {"PMR": rg_group({grabbed.url: grabbed}), "Bad": rg_group({failed.url: failed})}
        )

        assert [f.name for f in embed.fields[:2]] == ["Grabbed · PMR", "Failed · Bad"]

    def test_nothing_added_pushes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._grab(monkeypatch, FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show-PMR")})) == []

    def test_dry_run_never_pushes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A dry run simulates the add (so the count is non-zero) but the push is
        # an outward notification and must stay silent.
        assert self._grab(monkeypatch, FakeTorrents({"h1": (AddStatus.ADDED, "Show-PMR")}), qbit=None) == []
