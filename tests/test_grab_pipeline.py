# pyright: strict
# pyright: reportPrivateUsage=false
# The add-path assertions read the pipeline's private wiring (_grab / _state), which
# strict re-flags. The repo disables reportPrivateUsage for tests.
"""Unit tests for the grab "produce" side (`GrabPipeline`).

Pin the add path - `_grab_release` registering durable `PendingImport`
rows, `grab_releases`'s cap bookkeeping, and `_grab` returning the
added count (it never finalizes. The arr run loop owns the single finalize site).
Built bare (`object.__new__` via `make_bare_instance`) so no live qBittorrent
login happens. The client `add` is faked by `FakeTorrents`.
"""

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
import qbittorrentapi
from seadex import Tracker

from pearlarr import notify
from pearlarr.arr_run_state import ArrRunState, PerTitleState
from pearlarr.config import Arr
from pearlarr.discord import DiscordEmbed
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_pipeline import (
    _SKIP_REPORTS,
    NO_PENDING_IMPORT_UPDATES,
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
    install_hub,
    severity_of,
)
from pearlarr.pending_import import OwnedAtGrab, PendingImport
from pearlarr.releases import AnotherTitlesCounts, Release, SeadexDict, SkipReason
from pearlarr.stamps import now_stamp, stamp_of
from pearlarr.torrent_service import AddResult, GrabResult, TorrentAddError
from pearlarr.tracker_links import TrackerPageError
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    CLIENT_SENTINEL,
    PENDING_IMPORT_AL_ID,
    AddStatus,
    FakeCacheStore,
    FakeTorrents,
    claim_al_ids,
    grab_request,
    make_entry_record,
    make_grab_pipeline,
    make_release,
    one_release_dict,
    pending_import,
    pending_import_update,
    rg_group,
)
from .fakes import RecordingHub, diagnostic_messages, install_recording_hub


def _stub_add_torrent(req: GrabRequest) -> tuple[int, list[GrabResult]]:
    """Replaces `GrabPipeline.grab_releases` for the cap-notice tests.

    Returns a fixed `(n_added, results)` so `_grab`'s cap notice is exercised without a real qBittorrent add. The
    counter it reads (`_state.torrents_added`) is set by the test.
    """

    del req
    return 1, []


def _pipeline(
    *,
    torrents: FakeTorrents,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config: object,
) -> GrabPipeline:
    """A bare `GrabPipeline` wired for the add path (a blocking run, not a dry run)."""

    return make_grab_pipeline(
        _torrents=torrents,
        qbit=qbit,
        _state=ArrRunState(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode),
        **config,
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


class TestGrabAnnouncesTheCap:
    """`_grab` returns the added count and posts the cap notice once, from the title whose adds crossed it.

    GrabPipeline holds no reference back to the arr run loop, so the notice is the only cap signal it emits: the scan
    never stops on it.
    """

    @staticmethod
    def _request() -> GrabRequest:
        return grab_request(entry=make_entry_record(url="https://seadex.example/1"))

    def test_the_title_crossing_the_cap_announces_it(self) -> None:
        # The stub reports one add without touching the counter, so the counter is advanced by hand to the cap.
        recording = install_recording_hub()
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)

        def add_and_count(req: GrabRequest) -> tuple[int, list[GrabResult]]:
            state.torrents_added += 1
            return _stub_add_torrent(req)

        pipeline = make_grab_pipeline(max_torrents_to_add=1, grab_releases=add_and_count, _state=state)
        # Warm the gateway cache so the embed's thumb lookup never hits AniList.
        pipeline._anilist.al_cache.update({1: {}})

        assert pipeline._grab(self._request()) == 1
        assert [e.cap for e in recording.of_type(CapReached)] == [1]

    def test_a_title_already_past_the_cap_stays_quiet(self) -> None:
        # The notice belongs to the crossing title alone: a later title adds nothing and re-announces nothing.
        recording = install_recording_hub()
        pipeline = make_grab_pipeline(max_torrents_to_add=1, grab_releases=_stub_add_torrent)
        pipeline._anilist.al_cache.update({1: {}})
        pipeline._state.torrents_added = 1  # already at the cap of 1

        pipeline._grab(self._request())

        assert recording.of_type(CapReached) == []

    def test_a_dry_run_never_announces_the_cap(self) -> None:
        # MUTATION PIN: reading the raw config cap instead of _effective_cap would announce a cap a dry run
        # never enforces.
        recording = install_recording_hub()
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)

        def add_and_count(req: GrabRequest) -> tuple[int, list[GrabResult]]:
            state.torrents_added += 1
            return _stub_add_torrent(req)

        pipeline = make_grab_pipeline(qbit=None, max_torrents_to_add=1, grab_releases=add_and_count, _state=state)
        pipeline._anilist.al_cache.update({1: {}})

        pipeline._grab(self._request())

        assert recording.of_type(CapReached) == []


class TestGrabPushesNotice:
    """`_grab` builds a `GrabNotice` from the request and add outcomes, then pushes it.

    It never pushes in a dry run, and never when nothing was actually added.
    """

    def _grab(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        outcome: AddStatus = AddStatus.ADDED,
        qbit: object = CLIENT_SENTINEL,
    ) -> list[DiscordEmbed]:
        embeds: list[DiscordEmbed] = []

        def record(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
            del url, client
            embeds.append(embed)

        monkeypatch.setattr(notify, "discord_push", record)
        pipeline = _pipeline(
            torrents=FakeTorrents({"h1": (outcome, "Show-PMR")}),
            qbit=qbit,
            _notifier=Notifier(
                discord_url="https://discord.example",
                webhook_url=None,
                web=httpx.Client(),
            ),
        )
        # Warm the gateway cache so the art lookups never hit AniList.
        pipeline._anilist.al_cache.update(
            {
                7: {
                    "data": {
                        "Media": {
                            "coverImage": {"large": "https://img/cover"},
                            "bannerImage": "https://img/banner",
                        },
                    },
                },
            },
        )
        pipeline._grab(
            GrabRequest(
                al_id=7,
                # The library item's name must not prefix the AniList one, or the byline
                # this class asserts on would (correctly) dedupe away.
                arr_title="The Show",
                title_name="Show Title",
                entry=make_entry_record(url="https://releases.moe/7", notes="the why"),
                seadex_dict=one_release_dict(srg="PMR", infohash="h1"),
                update={},
                replaced_groups=("OldGroup",),
                coverage="S01 E01-E12",
            ),
        )
        return embeds

    def test_added_pushes_the_resolved_notice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        [embed] = self._grab(monkeypatch)

        # The request's entry/coverage and the gateway's art all reached the embed.
        assert embed.title == "Show Title"
        assert embed.url == "https://releases.moe/7"
        assert embed.thumb_url == "https://img/cover"
        assert embed.image_url == "https://img/banner"
        # A single-group grab hoists its release into the description. The
        # subtitle/notes stack trails as the nameless (header-free) field.
        assert embed.description == "**Grabbed · `PMR`**\n[Nyaa](https://nyaa.si/view/1)"
        assert [f.name for f in embed.fields] == ["Episodes", "Replacing", ""]
        assert embed.fields[-1].value == "-# The Show\n> the why"

    def test_nothing_added_pushes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._grab(monkeypatch, outcome=AddStatus.ALREADY_IN_QBIT) == []

    def test_dry_run_never_pushes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A dry run simulates the add (so the count is non-zero) but the push is
        # an outward notification and must stay silent.
        assert self._grab(monkeypatch, qbit=None) == []


class TestGrabReleaseRegistersPending:
    """`_grab_release` persists a pending import from its update for both a fresh and an already-present torrent.

    It records a grab and counts toward the cap only for a fresh add.
    """

    def test_already_in_qbit_registers_pending_import(self) -> None:
        # The preferred release is already in qBittorrent with no pending import (grabbed
        # before the store had it, or its row dropped): track it for the wait pass,
        # but never count it as a this-run grab.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        facts = OwnedAtGrab(entry_groups=("NAN0",))
        updates = {"h1": pending_import_update("h1", guards=facts)}

        n_added, results = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert claim_al_ids(_stored(pipeline, "h1")) == (PENDING_IMPORT_AL_ID,)
        # A registration writes the title's guard row too: evidence follows the
        # newest plan, never a frozen per-pending-import copy.
        assert _guards(pipeline) == {PENDING_IMPORT_AL_ID: facts}
        assert pipeline._state.reacquired_keys == {"h1"}
        assert pipeline._state.grabbed_this_run == {}
        assert n_added == 0
        assert pipeline._state.torrents_added == 0
        assert pipeline._state.stats.grabbed == []
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT]

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        facts = OwnedAtGrab(entry_groups=("NAN0",), stale_groups=("OldPick",))
        updates = {"h1": pending_import_update("h1", guards=facts)}

        n_added, _ = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert set(_pending_import_rows(pipeline)) == {"h1"}
        assert _guards(pipeline) == {PENDING_IMPORT_AL_ID: facts}
        assert [p.infohash for p in pipeline._state.grabbed_this_run.values()] == ["h1"]
        assert n_added == 1
        assert pipeline._state.torrents_added == 1
        assert len(pipeline._state.stats.grabbed) == 1

    def test_already_in_qbit_does_not_count_toward_cap(self) -> None:
        # One already-present + one fresh, cap 1: only the fresh add counts.
        torrents = FakeTorrents(
            {
                "already": (AddStatus.ALREADY_IN_QBIT, "old"),
                "fresh": (AddStatus.ADDED, "new"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        seadex_dict = {
            **one_release_dict(srg="OLD", infohash="already", url="https://nyaa.si/view/1"),
            **one_release_dict(srg="NEW", infohash="fresh", url="https://nyaa.si/view/2"),
        }
        updates = {"already": pending_import_update("already"), "fresh": pending_import_update("fresh")}

        n_added, _ = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict, pending_import_updates=updates))

        assert n_added == 1
        assert pipeline._state.torrents_added == 1

    def test_a_pending_import_update_built_without_the_stored_pending_import_replaces_it(self) -> None:
        # The update's own `pending_import` decides the branch, never the store: a second title's update built
        # without the pending import reacquires and overwrites it (one pending import per torrent, the last writer's).
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")})
        pipeline = _pipeline(torrents=torrents)
        first = pending_import_update("h1", al_id=11, title="Cour 1")
        second = pending_import_update("h1", al_id=22, title="Cour 2")

        pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={"h1": first})
        )
        pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={"h1": second})
        )

        assert set(_pending_import_rows(pipeline)) == {"h1"}
        assert pipeline._state.reacquired_keys == {"h1"}
        assert claim_al_ids(_stored(pipeline, "h1")) == (22,)

    def test_a_second_pending_import_update_carrying_the_first_pending_import_joins_its_claim(self) -> None:
        # Two titles share one torrent in one run: A's add tracks the pending import fresh, B's dedups to
        # ALREADY_IN_QBIT with an update carrying A's pending import (the store read). B's claim joins it, the
        # run-list copy follows, and a this-run grab is never a reacquire.
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)
        first = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}), cache_store=store, _state=state
        )
        first.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_import_updates={"h1": pending_import_update("h1", al_id=11, title="Cour 1")},
            )
        )
        second = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), cache_store=store, _state=state
        )
        update = pending_import_update("h1", al_id=22, title="Cour 2", stored=_stored(first, "h1"))

        second.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={"h1": update})
        )

        assert claim_al_ids(_stored(second, "h1")) == (11, 22)
        assert claim_al_ids(state.grabbed_this_run["h1"]) == (11, 22)
        assert state.reacquired_keys == set()
        assert _guards(second).keys() == {11, 22}

    def test_no_pending_import_update_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "x")})
        pipeline = _pipeline(torrents=torrents)

        pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates={}),
        )

        assert _pending_import_rows(pipeline) == {}
        assert pipeline._state.grabbed_this_run == {}

    def test_off_mode_registers_nothing_off_the_strategys_empty_pending_import_updates(self) -> None:
        # The wait-mode gate is the strategy's: it builds no pending import updates when the mode is off, and the
        # pipeline persists only what it is fed.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "x")})
        pipeline = _pipeline(torrents=torrents, mode=ImportWaitMode.OFF)

        pipeline.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_import_updates=NO_PENDING_IMPORT_UPDATES,
            ),
        )

        assert _pending_import_rows(pipeline) == {}
        assert _guards(pipeline) == {}
        assert pipeline._state.grabbed_this_run == {}

    def test_dry_run_does_not_register_but_returns_outcome(self) -> None:
        # No client -> dry run: nothing persisted, but the outcome still surfaces.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "x")})
        pipeline = _pipeline(torrents=torrents, qbit=None)
        updates = {"h1": pending_import_update("h1")}

        _, results = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert _pending_import_rows(pipeline) == {}
        assert _guards(pipeline) == {}
        assert pipeline._state.grabbed_this_run == {}
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
    """`_register_pending_import`'s three branches: a fresh add tracks, an ALREADY_IN_QBIT joins or reacquires.

    The update's `pending_import` decides between the last two, never the store. Only an update with no
    stored pending import is stamped at qBittorrent's add time.
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
        pipeline = _pipeline(torrents=self._reacquire(datetime.now() - timedelta(days=100)))
        resident = self._resident(pipeline)

        self._add(pipeline, pending_import_update("h1", al_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert pending.added_at == self._STORED_AT
        assert [c.al_id for c in pending.claims] == [PENDING_IMPORT_AL_ID, 22]
        assert pipeline._state.reacquired_keys == {"h1"}
        assert pipeline._state.grabbed_this_run == {}

    def test_this_runs_own_grab_is_never_a_reacquire(self) -> None:
        # One entry listing a torrent twice: the second add meets this run's new pending import, and the
        # torrent stays the run's grab.
        pipeline = _pipeline(torrents=self._reacquire(None))
        pipeline._state.grabbed_this_run["h1"] = pending_import(infohash="h1")

        self._add(pipeline, pending_import_update("h1"))

        assert pipeline._state.reacquired_keys == set()

    def test_resident_reacquire_refreshes_the_guard_row(self) -> None:
        # The title's guard evidence follows the newest plan, never a frozen copy:
        # a claim joining a carried-over pending import re-puts the row the trust read loads.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        facts = OwnedAtGrab(entry_groups=("NAN0",))

        self._add(pipeline, pending_import_update("h1", guards=facts, stored=resident))

        assert _guards(pipeline) == {PENDING_IMPORT_AL_ID: facts}

    def test_a_joining_title_writes_only_its_own_claims_guard_row(self) -> None:
        # The stored claim's row is the evidence its own grab wrote. A second title joining the pending import
        # never re-puts it from the loaded copy, so a row written since the copy was read stands.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        newer = OwnedAtGrab(entry_groups=("NewPick",))
        joining = OwnedAtGrab(entry_groups=("NAN0",))
        pipeline.cache_store.put_guards(Arr.SONARR, PENDING_IMPORT_AL_ID, newer)

        self._add(pipeline, pending_import_update("h1", al_id=22, guards=joining, stored=resident))

        assert _guards(pipeline) == {PENDING_IMPORT_AL_ID: newer, 22: joining}

    def test_a_joining_title_replaces_its_own_claim_and_merges_the_file_map(self) -> None:
        # A new grab by the title already claiming the pending import: its claim is replaced and
        # the new placements merge into the file map.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        update = pending_import_update(
            "h1", ordered_episode_ids=(101, 102), placements={"Show - 02 [1080p].mkv": (102,)}, stored=resident
        )

        self._add(pipeline, update)

        pending = _stored(pipeline, "h1")
        (claim,) = pending.claims
        assert claim.ordered_episode_ids == (101, 102)
        assert dict(pending.file_episode_map) == {
            normalize_file_name("Show - 01 [1080p].mkv"): (101,),
            normalize_file_name("Show - 02 [1080p].mkv"): (102,),
        }

    def test_a_new_pending_import_with_no_add_time_starts_now(self) -> None:
        # No stored pending import and no usable qBittorrent time: tracked from now, with the
        # title's guard row.
        pipeline = _pipeline(torrents=self._reacquire(None))
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1"))

        pending = _stored(pipeline, "h1")
        assert before <= pending.added_at <= now_stamp()
        assert pipeline._state.reacquired_keys == {"h1"}
        assert pipeline._state.grabbed_this_run == {}
        assert _guards(pipeline).keys() == {PENDING_IMPORT_AL_ID}

    def test_a_torrent_long_in_qbittorrent_is_tracked_at_its_add_time(self) -> None:
        # No stored pending import: the new one starts at qBittorrent's add time. A torrent's age blocks nothing
        # and warns of nothing.
        added_on = datetime.now() - timedelta(days=100)
        pipeline = _pipeline(torrents=self._reacquire(added_on))
        recording = install_recording_hub()

        results = self._add(pipeline, pending_import_update("h1"))

        assert diagnostic_messages(recording, Severity.WARNING) == []
        assert _stored(pipeline, "h1").added_at == stamp_of(added_on)
        assert pipeline._state.reacquired_keys == {"h1"}
        assert pipeline._state.grabbed_this_run == {}
        assert _guards(pipeline).keys() == {PENDING_IMPORT_AL_ID}
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT]

    def test_a_marked_release_is_tracked_under_its_cross_seed(self) -> None:
        # The torrent that already has the release's files is tracked like one under SeaDex's hash.
        pipeline = _pipeline(torrents=self._reacquire(datetime.now() - timedelta(days=3)))
        seadex_dict = one_release_dict(srg="NAN0", infohash="h1")
        seadex_dict["NAN0"].releases["https://nyaa.si/view/1"].cross_seed_hash = "c1"

        pipeline.grab_releases(
            grab_request(seadex_dict=seadex_dict, pending_import_updates={"c1": pending_import_update("c1")})
        )

        assert _pending_import_rows(pipeline).keys() == {"c1"}

    def test_fresh_add_never_joins_reacquired_keys(self) -> None:
        # The fresh-add branch is outcome-keyed: an ADDED is a this-run grab, stamped now.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}))
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1"))

        assert pipeline._state.reacquired_keys == set()
        assert [p.infohash for p in pipeline._state.grabbed_this_run.values()] == ["h1"]
        assert before <= _stored(pipeline, "h1").added_at <= now_stamp()

    def test_fresh_add_of_a_stored_pending_import_restamps_the_start(self) -> None:
        # A re-add of a torrent whose pending import survived (qBittorrent lost it): the
        # joined pending import enters the run list restarted at now, the start a new
        # download's rescan proof counts from.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}))
        resident = self._resident(pipeline)
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1", al_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert [c.al_id for c in pending.claims] == [PENDING_IMPORT_AL_ID, 22]
        assert before <= pending.added_at <= now_stamp()
        assert pipeline._state.reacquired_keys == set()
        assert list(pipeline._state.grabbed_this_run) == ["h1"]


def _nyaa_release(*, url: str, infohash: str) -> Release:
    """A Nyaa release marked to grab that clears every add-path filter."""

    item = make_release(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return item


class TestAddTorrentCap:
    """add_torrent honors max_torrents_to_add within ONE title's release loop."""

    def test_cap_holds_the_releases_past_it(self) -> None:
        # MUTATION PIN: `cap = None` and `>= cap` -> `> cap` both over-grab. Three
        # releases to grab under a cap of 2: exactly the first two reach the service,
        # both counters read 2, and the third release is never attempted.
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
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=2)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2"]  # h3 is held, never asked of qBittorrent
        assert n_added == 2
        assert pipeline._state.torrents_added == 2
        assert [r.outcome for r in results] == [AddStatus.ADDED, AddStatus.ADDED]
        assert pipeline._state.per_title.held_by_cap is True
        assert pipeline._state.stats.held_by_cap == 1

    def test_a_title_hitting_the_cap_exactly_holds_nothing(self) -> None:
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "one")}), max_torrents_to_add=1)

        n_added, _results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert n_added == 1
        assert pipeline._state.per_title.held_by_cap is False
        assert pipeline._state.stats.held_by_cap == 0

    def test_a_title_past_the_cap_is_held_and_tallied_once(self) -> None:
        # Two grabbable releases past the cap: one hold flag, one tally, no qBittorrent call.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one"), "h2": (AddStatus.ADDED, "two")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._state.torrents_added = 1

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == []
        assert (n_added, results) == (0, [])
        assert pipeline._state.per_title.held_by_cap is True
        assert pipeline._state.stats.held_by_cap == 1

    def test_a_screened_out_release_past_the_cap_is_not_held(self) -> None:
        # The private skip still lands (its flag and line), and no hold forms for a release that could not be grabbed.
        private = make_release(url="https://private.example/1", infohash="h1", download=True, is_public=False)
        seadex_dict: SeadexDict = {"RG": rg_group({private.url: private})}
        pipeline = _pipeline(torrents=FakeTorrents({}), max_torrents_to_add=1)
        pipeline._state.torrents_added = 1

        pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert pipeline._state.per_title.private_only_skipped is True
        assert pipeline._state.per_title.held_by_cap is False
        assert pipeline._state.stats.held_by_cap == 0

    def test_a_joining_title_held_by_the_cap_saves_its_claim(self) -> None:
        # A torrent already in qBittorrent under a stored pending import stays tracked past the cap: the title's claim
        # joins the pending import without a qBittorrent call, and nothing joins the reacquired keys.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._state.torrents_added = 1
        resident = pending_import(infohash="h1", al_id=11)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", al_id=22, stored=resident)}

        pipeline.grab_releases(grab_request(seadex_dict=seadex_dict, pending_import_updates=updates))

        assert torrents.calls == []
        assert claim_al_ids(_stored(pipeline, "h1")) == (11, 22)
        assert pipeline._state.reacquired_keys == set()
        assert pipeline._state.grabbed_this_run == {}

    def test_a_held_pending_import_update_for_a_new_torrent_stores_nothing(self) -> None:
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "one")}), max_torrents_to_add=1)
        pipeline._state.torrents_added = 1

        pipeline.grab_releases(
            grab_request(seadex_dict=seadex_dict, pending_import_updates={"h1": pending_import_update("h1")})
        )

        assert pipeline.cache_store.get_pending_import_row(Arr.SONARR, "h1") is None
        assert pipeline._state.per_title.held_by_cap is True

    def test_zero_removes_the_cap(self) -> None:
        # MUTATION PIN: without the `cap == 0` branch, 0 would flow into `>= cap`
        # and stop the run after the FIRST add instead of never.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents(
            {
                "h1": (AddStatus.ADDED, "one"),
                "h2": (AddStatus.ADDED, "two"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=0)

        n_added, _results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2"]
        assert n_added == 2

    def test_dry_run_ignores_the_cap(self) -> None:
        # A dry run must walk the whole library so its report stays complete:
        # three would-be adds under a cap of 2 all go through, and no stop fires.
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
        pipeline = _pipeline(torrents=torrents, qbit=None, max_torrents_to_add=2)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2", "h3"]
        assert n_added == 3
        assert [r.outcome for r in results] == [AddStatus.ADDED] * 3

    def test_non_added_release_does_not_stop_the_loop(self) -> None:
        # MUTATION PIN: the non-ADDED `continue` flipped to `break` would abandon
        # the rest of the group's releases. ALREADY_IN_QBIT first, ADDED second, ONE
        # group: both must be attempted, in order.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="already")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="fresh")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents(
            {
                "already": (AddStatus.ALREADY_IN_QBIT, "old"),
                "fresh": (AddStatus.ADDED, "new"),
            }
        )
        pipeline = _pipeline(torrents=torrents)

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["already", "fresh"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT, AddStatus.ADDED]


class TestGrabAndCacheAtTheCap:
    """The title whose add hits the cap exactly is cached; a title with a held release is not, and still paces."""

    @staticmethod
    def _request(al_id: int, seadex_dict: SeadexDict) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=seadex_dict,
            update={"updated_at": "2026-01-01 00:00:00"},
        )

    def test_hitting_the_cap_exactly_caches_the_title(self) -> None:
        # MUTATION PIN: a "cap reached" condition in the cache gate would leave the crossing title uncached, so the
        # next run re-checked and re-grabbed it. Drive the real add path to the cap.
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1")))

        assert pipeline._state.torrents_added == 1
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is not None
        assert pipeline._state.stats.needs_action == []

    def test_a_held_title_is_neither_cached_nor_a_needs_action_row(self) -> None:
        # Held is not a problem to act on: the title stays uncached so the next run grabs it.
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"
        pipeline._state.torrents_added = 1

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1")))

        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert pipeline._state.stats.needs_action == []
        assert pipeline._state.stats.held_by_cap == 1

    def test_a_held_title_posts_the_held_status(self) -> None:
        recording = install_recording_hub()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), max_torrents_to_add=1)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.torrents_added = 1

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1")))

        statuses = [d.value.text for d in recording.of_type(TitleDetail) if d.label == "status"]
        assert statuses == ["held by the run cap; grabbed next run"]


class TestUpToDateTally:
    """The up-to-date counter accumulates across titles."""

    def test_two_up_to_date_titles_both_counted(self) -> None:
        # MUTATION PIN: `stats.up_to_date += 1` degraded to `= 1` clamps at one.
        # Two nothing-to-grab titles must tally 2.
        pipeline = _pipeline(torrents=FakeTorrents({}), sleep_time=0)

        for al_id in (1, 2):
            pipeline.grab_and_cache(
                grab_request(al_id=al_id, entry=make_entry_record(url=f"https://seadex.example/{al_id}"))
            )

        assert pipeline._state.stats.up_to_date == 2


def _anidex_release(*, url: str, infohash: str) -> Release:
    """A release marked to grab on AniDex: public (clears the private-only gate), in the default tracker set.

    It has no parser, so it hits `_grab_release`'s new skip.
    """

    item = make_release(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.ANIDEX
    return item


class TestUnsupportedTrackerSkip:
    """An unparseable tracker is skipped (not raised), so the title's other releases still grab.

    A title with nothing grabbable is left uncached and flagged.
    """

    def test_skipped_but_loop_continues(self) -> None:
        # AniDex first, Nyaa second, under one group. The old raise unwound the whole
        # release loop - dropping the grabbable Nyaa release too. Now AniDex is skipped and
        # the loop continues. Default config: private_releases warn, all trackers selected.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        nyaa = make_release(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddStatus.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn")
        updates = {"hn": pending_import_update("hn")}

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict, pending_import_updates=updates))

        # AniDex never reached the service. Only Nyaa was handed over and added.
        assert torrents.calls == ["hn"]
        assert n_added == 1
        assert pipeline._state.torrents_added == 1
        assert [r.outcome for r in results] == [AddStatus.ADDED]
        assert pipeline._state.per_title.unsupported_tracker_skipped is True
        assert pipeline._state.per_title.unsupported_tracker_groups == ["NAN0"]

    def test_unsupported_only_title_left_uncached_and_listed(self) -> None:
        # The title's only release is on AniDex: nothing grabbable, so the title must
        # NOT be cached (re-checked next run) and surfaces once in needs-action.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        # Warm the AniList cache so _grab's thumbnail lookup stays offline.
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=42,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            update={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)
        assert pipeline._state.torrents_added == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [r.reason for r in pipeline._state.stats.needs_action] == ["tracker not yet supported; grab manually"]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.UNSUPPORTED_TRACKER]

    def test_private_and_unsupported_surfaces_only_private(self) -> None:
        # Both a private-only skip AND an unsupported-tracker skip on one title,
        # nothing grabbed: exactly ONE needs-action reason (private-only wins) - the
        # two reasons are either/or, never both.
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        seadex_dict: SeadexDict = {"NAN0": rg_group({private.url: private, anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            update={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        # Both skips happened...
        assert pipeline._state.per_title.private_only_skipped is True
        assert pipeline._state.per_title.unsupported_tracker_skipped is True
        # ...but only the private-only reason is surfaced, and the title stays uncached.
        assert [r.reason for r in pipeline._state.stats.needs_action] == [
            "private-only release; private releases not supported"
        ]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 7) is None

    def test_private_only_in_fallback_mode_surfaces_no_alternative(self) -> None:
        # private_releases: fallback and still nothing grabbable means no
        # fallback covered the title's files: the needs-action row says that
        # (its own cause, so the summary tip doesn't suggest the fallback that's
        # already on).
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            update={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert pipeline._state.per_title.private_only_skipped is True
        assert [r.reason for r in pipeline._state.stats.needs_action] == [
            "private-only release; no public alternative covers these files"
        ]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 7) is None

    def test_outdated_skip_in_fallback_mode_surfaces_the_outdated_cause(self) -> None:
        # The planner skipped a preferred release the arr has at an outdated size, which a fallback must not
        # replace (the outdated state bit comes in): the needs-action row gets its
        # own cause + reason, and the title stays uncached.
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"
        pipeline._state.per_title.private_only_outdated = True

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            update={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert pipeline._state.per_title.private_only_skipped is True
        assert [r.reason for r in pipeline._state.stats.needs_action] == [
            (
                "private-only release; your copy is outdated (its file size no longer matches) "
                "and only a fallback covers it"
            )
        ]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_OUTDATED]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 7) is None

    def test_interactive_private_pick_reads_as_a_hand_picked_no_fallback(self) -> None:
        # Interactive + fallback: a skip here is a hand-picked private release, so
        # the reason says so - but the cause stays PRIVATE_ONLY_NO_FALLBACK so the summary tip
        # never suggests enabling the fallback that's already on.
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            update={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert [r.reason for r in pipeline._state.stats.needs_action] == [
            "hand-picked private release; private releases not supported"
        ]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK]


class TestFallbackMarker:
    """A title's fallback marker is set when a fallback stood in for a private release, and cleared otherwise."""

    def test_fallback_grab_caches_the_title(self) -> None:
        # The fallback happy path: the planner already dropped the private release's grab
        # (public fallback kept), the fallback adds fine -> the title is cached
        # with no needs-action row, unlike warn mode's uncached skip.
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=False)
        private.tracker = Tracker.ANIMEBYTES
        fall = make_release(url="https://nyaa.si/view/9", infohash="hf", download=True, is_fallback=True)
        fall.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {
            "Priv": rg_group({private.url: private}),
            "Fall": rg_group({fall.url: fall}),
        }

        torrents = FakeTorrents({"hf": (AddStatus.ADDED, "Show-Fall")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=42,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            update={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)
        assert pipeline._state.torrents_added == 1
        assert pipeline._state.per_title.private_only_skipped is False
        cached = pipeline.cache_store.get_cached_title(Arr.SONARR, 42)
        assert cached is not None
        # A fallback grab marks the title, so a switch to warn mode re-checks it.
        assert cached.fallback_satisfied is True
        assert pipeline._state.stats.needs_action == []

    def test_warn_mode_grab_clears_a_preseeded_marker(self) -> None:
        # A prior fallback run left fallback_satisfied=True. A later genuine grab
        # recomputes False and clears it (the marker is always written - the
        # partial-merge upsert would otherwise preserve the old True forever).
        nyaa = make_release(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"Pub": rg_group({nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddStatus.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline.cache_store.update_cached_title(Arr.SONARR, 7, {"fallback_satisfied": True})
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            title_name="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            update={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        cached = pipeline.cache_store.get_cached_title(Arr.SONARR, 7)
        assert cached is not None
        assert cached.fallback_satisfied is False


def _grab_request(seadex_dict: SeadexDict, *, input_missing: bool = False) -> GrabRequest:
    """Title 42's request over `seadex_dict`, its one group among the failed reads when `input_missing`."""

    return grab_request(
        al_id=42,
        entry=make_entry_record(url="https://seadex.example/42"),
        seadex_dict=seadex_dict,
        update={"updated_at": "2026-01-01 00:00:00"},
        input_unread_groups=("RG",) if input_missing else (),
    )


def _on_title(pipeline: GrabPipeline) -> GrabPipeline:
    """The pipeline mid-run on title 42, its AniList answer warm."""

    pipeline._anilist.al_cache.update({42: {}})
    pipeline._state.per_title.title_name = "Show"
    return pipeline


class TestPlacementInputUnread:
    """A failed Sonarr read under a grab-time placement leaves the title uncached, with a retry row in the summary."""

    def test_a_grabbed_title_stays_uncached_with_a_retry_row(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict, input_missing=True))
        assert pipeline._state.torrents_added == 1
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = pipeline._state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]
        assert rows[0].reason == "a read the placement needs failed; will retry next run"
        assert rows[0].group == "RG"

    def test_a_title_with_nothing_to_grab_is_neither_up_to_date_nor_cached(self) -> None:
        # The placement may have left a sequence waiting, so the coverage decision was coarse: no "already have it".
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=False)})}
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict, input_missing=True))

        assert pipeline._state.stats.up_to_date == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]

    def test_a_failed_grab_outranks_the_unread_input(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict, input_missing=True))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]


def test_a_title_read_without_its_metadata_stays_uncached_with_nothing_to_do() -> None:
    # AniList or SkyHook was unreachable before the first fetch: the title re-checks next run, and no row asks for a
    # hand.
    seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=False)})}
    pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

    pipeline.grab_and_cache(replace(_grab_request(seadex_dict), metadata_unread=True))

    assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
    assert pipeline._state.stats.needs_action == []


def test_every_placement_skip_has_a_row() -> None:
    # The add loop reports a tracker skip itself. Any other reason missing from the table would leave its title
    # uncached with no needs-action row.
    tracker_skips = {SkipReason.PRIVATE_ONLY, SkipReason.UNSUPPORTED_TRACKER, SkipReason.TRACKER_NOT_SELECTED}
    assert _SKIP_REPORTS.keys() == set(SkipReason) - tracker_skips


class TestPlacementSkips:
    """A release the placement skipped is never grabbed: the title stays uncached, and the summary says why."""

    @staticmethod
    def _skipped_release(*, url: str, infohash: str, skip_reason: SkipReason) -> Release:
        """A Nyaa release marked to grab that the placement skipped for `skip_reason`."""

        item = _nyaa_release(url=url, infohash=infohash)
        item.skip_reason = skip_reason
        return item

    @pytest.mark.parametrize(
        ("skip_reason", "expected"),
        [
            (
                SkipReason.MISNUMBERED,
                SkipReport(
                    NeedsActionCause.MISNUMBERED,
                    "numbering does not match its SeaDex listing; import it by hand",
                ),
            ),
            (
                SkipReason.EPISODE_TITLES_CONTRADICT,
                SkipReport(
                    NeedsActionCause.EPISODE_TITLES_CONTRADICT,
                    "file titles contradict its numbering; import it by hand",
                ),
            ),
        ],
        ids=["misnumbered", "titles-refute"],
    )
    def test_a_torrent_left_for_a_hand_import_is_never_added_and_the_title_is_listed(
        self, skip_reason: SkipReason, expected: SkipReport
    ) -> None:
        recording = install_recording_hub()
        nyaa = self._skipped_release(url="https://nyaa.si/view/1", infohash="h1", skip_reason=skip_reason)
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert torrents.calls == []
        assert pipeline._state.torrents_added == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert pipeline._state.per_title.skipped_groups == {skip_reason: ["RG"]}
        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [skip_reason]
        (row,) = pipeline._state.stats.needs_action
        assert (row.cause, row.group, row.reason) == (expected.cause, "RG", expected.reason)

    def test_a_skipped_release_of_a_resident_torrent_never_joins_its_pending_import(self) -> None:
        # Another title's pending import has the torrent.
        resident = pending_import(infohash="h1")
        seadex_dict = one_release_dict(srg="RG", infohash="h1")
        seadex_dict["RG"].releases["https://nyaa.si/view/1"].skip_reason = SkipReason.EPISODE_TITLES_CONTRADICT
        torrents = FakeTorrents({"h1": AddResult(AddStatus.ALREADY_IN_QBIT, "Show", None)})
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, resident.infohash, resident.to_json())

        pipeline.grab_releases(
            grab_request(
                seadex_dict=seadex_dict,
                pending_import_updates={"h1": pending_import_update("h1", al_id=22, stored=resident)},
            )
        )

        assert torrents.calls == []
        assert [c.al_id for c in _stored(pipeline, "h1").claims] == [PENDING_IMPORT_AL_ID]

    def test_an_unread_listing_skips_the_release_until_the_next_run(self) -> None:
        # The release names its group among the reads that failed: the row is the retry row, not a hand import.
        recording = install_recording_hub()
        nyaa = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.PLACEMENT_INPUT_UNREAD
        )
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert pipeline._state.per_title.skipped_groups == {SkipReason.PLACEMENT_INPUT_UNREAD: ["RG"]}
        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [SkipReason.PLACEMENT_INPUT_UNREAD]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]

    def test_a_release_whose_files_are_another_titles_is_never_added_and_its_skip_carries_the_counts(self) -> None:
        recording = install_recording_hub()
        nyaa = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.PLACED_ELSEWHERE
        )
        nyaa.another_titles_counts = AnotherTitlesCounts(another_titles=12, unplaced=6)
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert pipeline._state.per_title.skipped_groups == {SkipReason.PLACED_ELSEWHERE: ["RG"]}
        assert [(e.reason, e.counts) for e in recording.of_type(ReleaseSkipped)] == [
            (SkipReason.PLACED_ELSEWHERE, AnotherTitlesCounts(another_titles=12, unplaced=6))
        ]
        (row,) = pipeline._state.stats.needs_action
        assert (row.cause, row.reason) == (
            NeedsActionCause.PLACED_ELSEWHERE,
            "its files place under another title SeaDex lists it with; any it could not place need a hand import",
        )

    def test_the_another_titles_row_outranks_a_failed_read(self) -> None:
        # A failed read clears next run. A release whose files are another title's stays skipped until the title has
        # them.
        another_titles = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.PLACED_ELSEWHERE
        )
        unread = self._skipped_release(
            url="https://nyaa.si/view/2", infohash="h2", skip_reason=SkipReason.PLACEMENT_INPUT_UNREAD
        )
        seadex_dict: SeadexDict = {
            "RG": rg_group({another_titles.url: another_titles}),
            "Other": rg_group({unread.url: unread}),
        }
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PLACED_ELSEWHERE]

    def test_a_failed_grab_outranks_the_another_titles_row(self) -> None:
        another_titles = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.PLACED_ELSEWHERE
        )
        failing = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {
            "RG": rg_group({another_titles.url: another_titles}),
            "Other": rg_group({failing.url: failing}),
        }
        torrents = FakeTorrents({}, raises={"h2": httpx.ConnectError("nyaa down")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]

    def test_a_skipped_release_beside_a_grabbable_one_lets_the_other_grab(self) -> None:
        skipped = self._skipped_release(url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.MISNUMBERED)
        other = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({skipped.url: skipped}), "Other": rg_group({other.url: other})}
        torrents = FakeTorrents({"h2": (AddStatus.ADDED, "Show-Other")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert torrents.calls == ["h2"]
        assert pipeline._state.torrents_added == 1
        # The grabbed half never caches the title: the skipped torrent is marked to grab again until the arr has its
        # files.
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.MISNUMBERED]

    def test_a_skipped_release_past_the_cap_posts_its_skip_and_no_cap_hold(self) -> None:
        # The skip screens the release before the cap: its skip line lands, and no hold forms for it.
        recording = install_recording_hub()
        skipped = self._skipped_release(url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.MISNUMBERED)
        seadex_dict: SeadexDict = {"RG": rg_group({skipped.url: skipped})}
        pipeline = _pipeline(torrents=FakeTorrents({}), max_torrents_to_add=1, sleep_time=0)
        pipeline._state.torrents_added = 1

        n_added, results = pipeline.grab_releases(grab_request(seadex_dict=seadex_dict))

        assert (n_added, results) == (0, [])
        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [SkipReason.MISNUMBERED]
        assert pipeline._state.per_title.held_by_cap is False
        assert pipeline._state.stats.held_by_cap == 0

    def test_a_release_not_to_grab_with_a_skip_reason_is_not_skipped_and_the_title_caches(self) -> None:
        # The planner found the arr has the torrent's files (a hand import done): the skip is moot, the title is up to
        # date.
        recording = install_recording_hub()
        skipped = self._skipped_release(url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.MISNUMBERED)
        skipped.to_grab = False
        seadex_dict: SeadexDict = {"RG": rg_group({skipped.url: skipped})}
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert recording.of_type(ReleaseSkipped) == []
        assert pipeline._state.per_title.skipped_groups == {}
        assert pipeline._state.stats.up_to_date == 1
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is not None

    def test_an_unsupported_tracker_outranks_the_misnumbered_row(self) -> None:
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        nyaa = self._skipped_release(url="https://nyaa.si/view/2", infohash="h1", skip_reason=SkipReason.MISNUMBERED)
        seadex_dict: SeadexDict = {"RG": rg_group({anidex.url: anidex, nyaa.url: nyaa})}
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.UNSUPPORTED_TRACKER]

    def test_the_misnumbered_row_outranks_a_failed_grab(self) -> None:
        skipped = self._skipped_release(url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.MISNUMBERED)
        failing = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({skipped.url: skipped}), "Other": rg_group({failing.url: failing})}
        torrents = FakeTorrents({}, raises={"h2": httpx.ConnectError("nyaa down")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.MISNUMBERED]

    def test_the_misnumbered_row_outranks_the_titles_contradict_row(self) -> None:
        # The table's order decides between two hand-import skips.
        contradicted = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.EPISODE_TITLES_CONTRADICT
        )
        misnumbered = self._skipped_release(
            url="https://nyaa.si/view/2", infohash="h2", skip_reason=SkipReason.MISNUMBERED
        )
        seadex_dict: SeadexDict = {
            "RG": rg_group({contradicted.url: contradicted}),
            "Other": rg_group({misnumbered.url: misnumbered}),
        }
        pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.MISNUMBERED]

    def test_the_titles_contradict_row_outranks_a_failed_grab(self) -> None:
        skipped = self._skipped_release(
            url="https://nyaa.si/view/1", infohash="h1", skip_reason=SkipReason.EPISODE_TITLES_CONTRADICT
        )
        failing = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({skipped.url: skipped}), "Other": rg_group({failing.url: failing})}
        torrents = FakeTorrents({}, raises={"h2": httpx.ConnectError("nyaa down")})
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(seadex_dict))

        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.EPISODE_TITLES_CONTRADICT]


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent unreachable/erroring) is contained at the add.

    ONE clean warning (no traceback), the release loop moves on, the title stays
    uncached (retried next run), and the summary carries a GRAB_FAILED
    needs-action row instead of the title silently vanishing.
    """

    def _request(self, al_id: int, seadex_dict: SeadexDict) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=seadex_dict,
            update={"updated_at": "2026-01-01 00:00:00"},
        )

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
        # Every boundary failure mode lands as ONE typed GrabFailed event (the
        # old path fell through to arr_run_loop's per-title traceback handler. The frozen
        # fact carries no traceback by construction, and it tallies WARNING).
        torrents = FakeTorrents({}, raises={"h1": error})
        pipeline = _pipeline(torrents=torrents)
        recording = RecordingHub()
        install_hub(recording.hub)

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
        assert pipeline._state.per_title.grab_failed_groups == ["RG"]

    def test_failed_only_title_stays_uncached_with_a_retry_row(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict))
        assert pipeline._state.torrents_added == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = pipeline._state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.GRAB_FAILED]
        assert rows[0].reason == "grab failed; will retry next run"
        assert rows[0].group == "RG"

    def test_partial_grab_with_a_failure_stays_uncached(self) -> None:
        # Like private_only_no_fallback: a failure blocks the cache even when a sibling
        # grabbed, so the failed release retries next run (the add dedups).
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddStatus.ADDED, "Show-RG")},
            raises={"hbad": qbittorrentapi.APIConnectionError("qbit died")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict))
        assert pipeline._state.torrents_added == 1
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]

    def test_cap_reached_with_a_failure_still_lands_the_grab_failed_row(self) -> None:
        # A title whose grab failed AND whose sibling add hit max_torrents_to_add
        # used to report nothing (the cap return skipped the needs-action tail):
        # the GRAB_FAILED row must still land, the run still stops, and the
        # cap-stopped title still isn't cached.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddStatus.ADDED, "Show-RG")},
            raises={"hbad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict))

        assert pipeline._state.torrents_added == 1
        rows = pipeline._state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.GRAB_FAILED]
        assert rows[0].group == "RG"
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None

    def test_next_clean_title_caches_after_a_failed_one(self) -> None:
        # The per-title failure note lives on PerTitleState, which begin_check
        # replaces per title: title 1's failure must not hold title 2's cache
        # write hostage.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents(
            {"h2": (AddStatus.ADDED, "Show-RG")},
            raises={"h1": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({1: {}, 2: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(1, {"RG": rg_group({bad.url: bad})}))
        # begin_check's new state between titles.
        pipeline._state.per_title = PerTitleState(title_name="Show S2")
        pipeline.grab_and_cache(self._request(2, {"RG": rg_group({good.url: good})}))

        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 1) is None
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 2) is not None


class TestFallbackModePrivateOnlySkipNeverCaches:
    """Fallback + non-interactive + a private-only skip: the title never caches, even on a partial grab.

    The no-fallback row resurfaces in every run's summary.
    """

    def _mixed_seadex_dict(self) -> SeadexDict:
        """A skipped private group next to a grabbable public group."""

        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        nyaa = make_release(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        return {"Priv": rg_group({private.url: private}), "Pub": rg_group({nyaa.url: nyaa})}

    def _request(self, al_id: int) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=self._mixed_seadex_dict(),
            update={"updated_at": "2026-01-01 00:00:00"},
        )

    def test_partial_grab_under_a_private_only_skip_stays_uncached_and_surfaces(self) -> None:
        # The public release adds fine while the private one is skipped: the fallback
        # couldn't cover the private files, so despite the grab the title must NOT
        # cache (re-checked next run) and the no-fallback row must land.
        torrents = FakeTorrents({"hn": (AddStatus.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42))
        assert pipeline._state.torrents_added == 1
        assert pipeline._state.per_title.private_only_skipped is True
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = pipeline._state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK]
        # Exact wording pinned once, in test_private_only_in_fallback_mode_surfaces_no_alternative.
        assert "no public alternative" in rows[0].reason

    def test_interactive_partial_grab_still_caches(self) -> None:
        # Interactive: the skip is a hand-picked private release, so
        # the plain gate stands - the partial grab caches the title as today.
        torrents = FakeTorrents({"hn": (AddStatus.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({43: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(43))

        assert pipeline._state.per_title.private_only_skipped is True
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 43) is not None
        assert pipeline._state.stats.needs_action == []


class TestShouldCacheAsDone:
    """The extracted cache predicate, pinned as a truth table.

    White-box: the state skip flags are set directly and the predicate is called
    with explicit gate inputs, so every blocking axis is pinned in isolation (the
    end-to-end grab_and_cache integration is pinned in the classes above).
    """

    @staticmethod
    def _predicate(
        *,
        private_releases: str = "warn",
        interactive: bool = False,
        private_only_skipped: bool = False,
        unsupported_tracker_skipped: bool = False,
        held_by_cap: bool = False,
        added_this_title: int = 0,
        grab_failed: bool = False,
        misnumbered: bool = False,
    ) -> bool:
        """One truth-table row. Every keyword is one axis of the predicate."""

        pipeline = make_grab_pipeline(private_releases=private_releases, interactive=interactive)
        pipeline._state.per_title.private_only_skipped = private_only_skipped
        pipeline._state.per_title.unsupported_tracker_skipped = unsupported_tracker_skipped
        pipeline._state.per_title.held_by_cap = held_by_cap
        if grab_failed:
            pipeline._state.per_title.grab_failed_groups.append("RG")
        if misnumbered:
            pipeline._state.per_title.add_skipped(SkipReason.MISNUMBERED, ("RG",))
        return pipeline._may_cache_title(added_this_title=added_this_title)

    def test_plain_grab_caches(self) -> None:
        assert self._predicate(added_this_title=1) is True

    def test_zero_added_with_nothing_skipped_caches(self) -> None:
        # The up-to-date case: nothing grabbed because nothing was needed.
        assert self._predicate() is True

    def test_a_placement_skip_keeps_the_title_uncached(self) -> None:
        # A release held past the cap is grabbed next run, so the title must re-check.
        assert self._predicate(added_this_title=1, held_by_cap=True) is False

    def test_a_private_only_skip_keeps_the_title_uncached_despite_a_partial_grab(self) -> None:
        # The documented-surprising row: fallback mode + non-interactive + a
        # private skip blocks caching even though something WAS grabbed - the
        # fallback couldn't cover the private files, so every run re-checks.
        assert self._predicate(private_releases="fallback", private_only_skipped=True, added_this_title=1) is False

    def test_a_grab_failure_keeps_the_title_uncached_despite_a_partial_grab(self) -> None:
        assert self._predicate(added_this_title=1, grab_failed=True) is False

    def test_a_misnumbered_torrent_keeps_the_title_uncached_despite_a_partial_grab(self) -> None:
        # The torrent is a human's: the title marks it to grab again every run until the arr has its files.
        assert self._predicate(added_this_title=1, misnumbered=True) is False

    def test_warn_mode_private_skip_keeps_the_title_cacheable(self) -> None:
        # A mixed grab in warn mode caches.
        assert self._predicate(private_only_skipped=True, added_this_title=1) is True

    def test_interactive_defuses_the_private_only_skip(self) -> None:
        # The skip is a hand-picked private release: plain gate stands.
        assert (
            self._predicate(
                private_releases="fallback",
                interactive=True,
                private_only_skipped=True,
                added_this_title=1,
            )
            is True
        )

    def test_zero_added_with_a_private_skip_stays_uncached(self) -> None:
        # Warn mode so the fallback-mode clause stays out: the skip clause alone keeps it uncached.
        assert self._predicate(private_only_skipped=True) is False

    def test_zero_added_with_an_unsupported_tracker_skip_stays_uncached(self) -> None:
        assert self._predicate(unsupported_tracker_skipped=True) is False

    def test_mixed_unsupported_tracker_grab_caches(self) -> None:
        # Something was grabbed, so the skipped tracker leaves the title cacheable.
        assert self._predicate(unsupported_tracker_skipped=True, added_this_title=1) is True
