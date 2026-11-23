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
from pearlarr.arr_run_state import ArrRunState
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
    install_hub,
    severity_of,
)
from pearlarr.pending_import import OwnedAtGrab, PendingImport
from pearlarr.releases import (
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
    FakeTorrents,
    arr_files,
    claim_anilist_ids,
    grab_request,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release,
    one_release_dict,
    pending_import,
    pending_import_update,
    rg_group,
    sonarr_ep,
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
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config: object,
) -> GrabPipeline:
    """A bare `GrabPipeline` wired for the add path (a blocking run, not a dry run)."""

    return make_grab_pipeline(
        _torrents=torrents,
        qbit=qbit,
        _state=ArrRunState(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=ImportWaitMode.BLOCKING),
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
        pipeline._anilist.answers.update({1: {}})

        assert pipeline._grab(self._request()) == 1
        assert [e.cap for e in recording.of_type(CapReached)] == [1]

    def test_a_title_already_past_the_cap_stays_quiet(self) -> None:
        # The notice belongs to the crossing title alone: a later title adds nothing and re-announces nothing.
        recording = install_recording_hub()
        pipeline = make_grab_pipeline(max_torrents_to_add=1, grab_releases=_stub_add_torrent)
        pipeline._anilist.answers.update({1: {}})
        pipeline._state.torrents_added = 1  # already at the cap of 1

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
        pipeline._anilist.answers.update(
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
                anilist_id=7,
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
    """`_grab_release` persists a pending import from its update, and a fresh add records a grab and counts toward the cap."""

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        facts = OwnedAtGrab(entry_groups=("NAN0",), stale_groups=("OldPick",))
        updates = {"h1": pending_import_update("h1", guards=facts)}

        n_added, _ = pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )

        assert set(_pending_import_rows(pipeline)) == {"h1"}
        assert _guards(pipeline) == {PENDING_IMPORT_ANILIST_ID: facts}
        assert [p.infohash for p in pipeline._state.grabbed_this_run.values()] == ["h1"]
        assert n_added == 1
        assert pipeline._state.torrents_added == 1
        assert len(pipeline._state.stats.grabbed) == 1

    def test_the_grabbed_row_names_the_placed_episodes_by_the_numbers_they_print_by(self) -> None:
        # Sonarr lists the aired S02E01 as S01E04.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")}))
        seadex = one_release_dict(srg="NAN0", infohash="h1")
        (release,) = seadex["NAN0"].releases.values()
        release.episodes = [PlacedEpisode(2, 1, 10, printed=PrintedKey(EpisodeKey(1, 4)))]

        pipeline.grab_releases(
            grab_request(seadex_dict=seadex, pending_import_updates={"h1": pending_import_update("h1")})
        )

        assert [row.coverage for row in pipeline._state.stats.grabbed] == ["S01 E04"]

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
                pending_import_updates={"h1": pending_import_update("h1", anilist_id=11, title="Cour 1")},
            )
        )
        second = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), cache_store=store, _state=state
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

        self._add(pipeline, pending_import_update("h1", anilist_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert pending.added_at == self._STORED_AT
        assert [c.anilist_id for c in pending.claims] == [PENDING_IMPORT_ANILIST_ID, 22]
        assert pipeline._state.reacquired_keys == {"h1"}
        assert pipeline._state.grabbed_this_run == {}

    def test_resident_reacquire_refreshes_the_guard_row(self) -> None:
        # The title's guard evidence follows the newest plan, never a frozen copy:
        # a claim joining a carried-over pending import re-puts the row the trust read loads.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        facts = OwnedAtGrab(entry_groups=("NAN0",))

        self._add(pipeline, pending_import_update("h1", guards=facts, stored=resident))

        assert _guards(pipeline) == {PENDING_IMPORT_ANILIST_ID: facts}

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
        assert _guards(pipeline).keys() == {PENDING_IMPORT_ANILIST_ID}
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT]

    def test_fresh_add_of_a_stored_pending_import_restamps_the_start(self) -> None:
        # A re-add of a torrent whose pending import survived (qBittorrent lost it): the
        # joined pending import enters the run list restarted at now, the start a new
        # download's rescan proof counts from.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}))
        resident = self._resident(pipeline)
        before = now_stamp()

        self._add(pipeline, pending_import_update("h1", anilist_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert [c.anilist_id for c in pending.claims] == [PENDING_IMPORT_ANILIST_ID, 22]
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

    def test_a_joining_title_held_by_the_cap_saves_its_claim(self) -> None:
        # A torrent already in qBittorrent under a stored pending import stays tracked past the cap: the title's claim
        # joins the pending import without a qBittorrent call, and nothing joins the reacquired keys.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._state.torrents_added = 1
        resident = pending_import(infohash="h1", anilist_id=11)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", anilist_id=22, stored=resident)}

        pipeline.grab_releases(grab_request(seadex_dict=seadex_dict, pending_import_updates=updates))

        assert torrents.calls == []
        assert claim_anilist_ids(_stored(pipeline, "h1")) == (11, 22)
        assert pipeline._state.reacquired_keys == set()
        assert pipeline._state.grabbed_this_run == {}


class TestGrabAndCacheAtTheCap:
    """A title with a release the cap held says so in its status."""

    @staticmethod
    def _request(anilist_id: int, seadex_dict: SeadexDict) -> GrabRequest:
        return grab_request(
            anilist_id=anilist_id,
            entry=make_entry_record(url=f"https://seadex.example/{anilist_id}"),
            seadex_dict=seadex_dict,
            update={"updated_at": "2026-01-01 00:00:00"},
        )

    def test_a_held_title_posts_the_held_status(self) -> None:
        recording = install_recording_hub()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), max_torrents_to_add=1)
        pipeline._anilist.answers.update({42: {}})
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

        for anilist_id in (1, 2):
            pipeline.grab_and_cache(
                grab_request(anilist_id=anilist_id, entry=make_entry_record(url=f"https://seadex.example/{anilist_id}"))
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
    """A release on an unparseable or private tracker is skipped (not raised).

    A title with nothing grabbable is left uncached and flagged.
    """

    def test_unsupported_only_title_left_uncached_and_listed(self) -> None:
        # The title's only release is on AniDex: nothing grabbable, so the title must
        # NOT be cached (re-checked next run) and surfaces once in needs-action.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        # Warm the AniList cache so _grab's thumbnail lookup stays offline.
        pipeline._anilist.answers.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            anilist_id=42,
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
        pipeline._anilist.answers.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            anilist_id=7,
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
        pipeline._anilist.answers.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            anilist_id=7,
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
            "private-only release; no fallback covers these files"
        ]
        assert [r.cause for r in pipeline._state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK]
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 7) is None

    def test_interactive_private_pick_reads_as_a_hand_picked_no_fallback(self) -> None:
        # Interactive + fallback: a skip here is a hand-picked private release, so
        # the reason says so - but the cause stays PRIVATE_ONLY_NO_FALLBACK so the summary tip
        # never suggests enabling the fallback that's already on.
        private = make_release(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.answers.update({7: {}})
        pipeline._state.per_title.title_name = "Show S1"

        req = GrabRequest(
            anilist_id=7,
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


def _grab_request(seadex_dict: SeadexDict, *, input_missing: bool = False) -> GrabRequest:
    """Title 42's request over `seadex_dict`, its one group among the failed reads when `input_missing`."""

    return grab_request(
        anilist_id=42,
        entry=make_entry_record(url="https://seadex.example/42"),
        seadex_dict=seadex_dict,
        update={"updated_at": "2026-01-01 00:00:00"},
        input_unread_groups=("RG",) if input_missing else (),
    )


def _on_title(pipeline: GrabPipeline) -> GrabPipeline:
    """The pipeline mid-run on title 42, its AniList answer warm."""

    pipeline._anilist.answers.update({42: {}})
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


def test_a_title_read_without_its_metadata_stays_uncached_with_nothing_to_do() -> None:
    # AniList or SkyHook was unreachable before the first fetch: the title re-checks next run, and no row asks for a
    # hand.
    seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=False)})}
    pipeline = _on_title(_pipeline(torrents=FakeTorrents({}), sleep_time=0))

    pipeline.grab_and_cache(replace(_grab_request(seadex_dict), metadata_unread=True))

    assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
    assert pipeline._state.stats.needs_action == []


class TestPlacementSkips:
    """A release the placement skipped is never grabbed: the title stays uncached, and the summary says why.

    One skipped over episodes Sonarr has no row for is never grabbed either, and its title is cached.
    """

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
                    "episode titles contradict its numbers; import it by hand",
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
        pipeline = _on_title(_pipeline(torrents=torrents, sleep_time=0))

        pipeline.grab_and_cache(_grab_request(plan.seadex_dict))

        assert nyaa.to_grab is True
        assert torrents.calls == []
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is not None
        assert pipeline._state.per_title.skipped_groups == {}
        assert [(e.reason, e.severity) for e in recording.of_type(ReleaseSkipped)] == [
            (SkipReason.AIRED_ONLY, Severity.INFO)
        ]
        assert pipeline._state.stats.needs_action == []


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent unreachable/erroring) is contained at the add.

    ONE clean warning (no traceback), the release loop moves on, the title stays
    uncached (retried next run), and the summary carries a GRAB_FAILED
    needs-action row instead of the title silently vanishing.
    """

    def _request(self, anilist_id: int, seadex_dict: SeadexDict) -> GrabRequest:
        return grab_request(
            anilist_id=anilist_id,
            entry=make_entry_record(url=f"https://seadex.example/{anilist_id}"),
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
        pipeline._anilist.answers.update({42: {}})
        pipeline._state.per_title.title_name = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict))
        assert pipeline._state.torrents_added == 0
        assert pipeline.cache_store.get_cached_title(Arr.SONARR, 42) is None
        rows = pipeline._state.stats.needs_action
        assert [r.cause for r in rows] == [NeedsActionCause.GRAB_FAILED]
        assert rows[0].reason == "grab failed; will retry next run"
        assert rows[0].group == "RG"
