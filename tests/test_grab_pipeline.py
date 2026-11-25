# pyright: strict
"""Tests for the grab side (`GrabPipeline`): what a title's grab adds, tracks, caches, posts, and lists.

`make_grab_pipeline` builds the pipeline bare, so no qBittorrent login happens. `FakeTorrents` stands in for
`TorrentService.add`: the tracker page read, the check for a torrent already in qBittorrent, and the add.
`tests/test_private_fallback.py` owns the fallback side of the cache gate, where a public release stands in for a
private one.
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from itertools import combinations
from typing import cast

import httpx
import pytest
import qbittorrentapi
import respx
from seadex import Tracker

from pearlarr.arr_run_state import ArrRunState, PerTitleState
from pearlarr.boundary import Json
from pearlarr.config import Arr, PrivateReleaseAction
from pearlarr.grab_pipeline import NO_PENDING_IMPORT_UPDATES, GrabPipeline, GrabRequest
from pearlarr.grab_placement import PendingImportUpdate
from pearlarr.notify import Notifier
from pearlarr.output import (
    CapReached,
    GrabAction,
    GrabbedRow,
    GrabFailed,
    GrabStatus,
    GroupLabel,
    NeedsActionCause,
    NeedsActionRow,
    ReleaseLabel,
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
    GroupReleases,
    PlacedEpisode,
    PrintedKey,
    Release,
    SeadexDict,
    SkipReason,
)
from pearlarr.stamps import now_stamp
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
    make_release_filter,
    make_services,
    pending_import,
    pending_import_update,
    placed_episode,
    rg_group,
    sonarr_ep,
)
from .fakes import RecordingHub, install_recording_hub

# Title 42 as its header sets it in a run, and the stamp its request caches it under.
_TITLE_URL = "https://seadex.example/42"
_TITLE_COVERAGE = "S01 E01-E12"
_UPDATED_AT = "2026-01-01 00:00:00"

_DISCORD_URL = "https://discord.example/hook"


def _run_state(*, torrents_added: int = 0, dry_run: bool = False) -> ArrRunState:
    """A blocking Sonarr run checking title 42, named Show, with `torrents_added` adds already counted toward the cap."""

    return ArrRunState(
        arr=Arr.SONARR,
        dry_run=dry_run,
        import_wait_mode=ImportWaitMode.BLOCKING,
        torrents_added=torrents_added,
        per_title=PerTitleState(title_name="Show", entry_url=_TITLE_URL, entry_coverage=_TITLE_COVERAGE),
    )


def _pipeline(
    *,
    torrents: FakeTorrents,
    state: ArrRunState | None = None,
    qbittorrent: bool = True,
    **overrides: object,
) -> GrabPipeline:
    """A bare `GrabPipeline` on `state` (a blocking run, not a dry run, when none is given).

    `qbittorrent` False leaves qBittorrent unset, which makes any run a dry run. `overrides` go to
    `make_grab_pipeline`: config settings, or a part of the pipeline to swap in.
    """

    qbit = CLIENT_SENTINEL if qbittorrent else None
    return make_grab_pipeline(_torrents=torrents, qbit=qbit, _state=state or _run_state(), **overrides)


def _discord() -> Notifier:
    """A notifier posting to the Discord webhook at `_DISCORD_URL`."""

    return Notifier(discord_url=_DISCORD_URL, webhook_url=None, web=httpx.Client())


def _notifying_pipeline(torrents: FakeTorrents) -> GrabPipeline:
    """A pipeline posting its grab notices to `_DISCORD_URL`, with AniList's cover and banner for title 42 alone."""

    return _pipeline(
        torrents=torrents,
        _notifier=_discord(),
        anilist_client=ScriptedAniListClient("Show Title", known=frozenset({42})),
    )


def _release(n: int, *, skipped: SkipReason | None = None) -> Release:
    """Release `n`, marked to grab: at `https://nyaa.si/view/<n>` with hash `h<n>`, or skipped for `skipped` when given.

    `SkipReason.PRIVATE_ONLY` puts it on AnimeBytes, private and with no hash, as SeaDex lists one.
    `SkipReason.UNSUPPORTED_TRACKER` puts it on AniDex, which has no parser. Any other reason is its placement's.
    """

    if skipped is SkipReason.PRIVATE_ONLY:
        url = f"https://animebytes.tv/torrents.php?id={n}"
        return make_release(url=url, tracker=Tracker.ANIMEBYTES, is_public=False, infohash=None, download=True)
    if skipped is SkipReason.UNSUPPORTED_TRACKER:
        return make_release(
            url=f"https://anidex.info/torrent/{n}", tracker=Tracker.ANIDEX, infohash=f"h{n}", download=True
        )
    release = make_release(url=f"https://nyaa.si/view/{n}", tracker=Tracker.NYAA, infohash=f"h{n}", download=True)
    release.skip_reason = skipped
    return release


def _group(*releases: Release) -> GroupReleases:
    """A release group's releases, keyed by url as SeaDex lists them."""

    return rg_group({release.url: release for release in releases})


def _grab_request(
    seadex_dict: SeadexDict,
    *,
    pending_import_updates: Mapping[str, PendingImportUpdate] = NO_PENDING_IMPORT_UPDATES,
    input_unread_groups: tuple[str, ...] = (),
) -> GrabRequest:
    """Title 42's request over `seadex_dict`, to be cached under `_UPDATED_AT`."""

    return grab_request(
        anilist_id=42,
        entry=make_entry_record(anilist_id=42, url=_TITLE_URL),
        seadex_dict=seadex_dict,
        update={"updated_at": _UPDATED_AT},
        pending_import_updates=pending_import_updates,
        input_unread_groups=input_unread_groups,
    )


def _tracked(pipeline: GrabPipeline) -> set[str]:
    """The hashes of the torrents the store holds a pending import for (what the arr run reads back)."""

    return set(pipeline.cache_store.get_pending_import_rows(Arr.SONARR))


def _stored(pipeline: GrabPipeline, infohash: str) -> PendingImport:
    """The store's pending import on the torrent, loaded (its stamps and claims need no guard row)."""

    raw = pipeline.cache_store.get_pending_import_row(Arr.SONARR, infohash)
    assert raw is not None
    return PendingImport.from_json(raw, guards={})


def _guards(pipeline: GrabPipeline) -> Mapping[int, OwnedAtGrab]:
    """The pipeline's durable per-title guard rows (what the reads load from)."""

    return pipeline.cache_store.get_guards(Arr.SONARR)


def _cached_at(pipeline: GrabPipeline, anilist_id: int = 42) -> str | None:
    """The stamp the title is cached under, or None when it is not cached."""

    cached = pipeline.cache_store.get_cached_title(Arr.SONARR, anilist_id)
    return None if cached is None else cached.updated_at


def _needs_action(group: str, reason: str, cause: NeedsActionCause) -> NeedsActionRow:
    """Title 42's needs-action row, its title, coverage and url as `_run_state` sets them."""

    return NeedsActionRow(
        title="Show", coverage=_TITLE_COVERAGE, group=group, url=_TITLE_URL, reason=reason, cause=cause
    )


def _statuses(recording: RecordingHub) -> list[str]:
    """The status lines posted under the title."""

    return [detail.value.text for detail in recording.of_type(TitleDetail) if detail.label == "status"]


def _posted_embed(webhook: respx.Route) -> dict[str, Json]:
    """The embed of the one notice posted to `webhook`."""

    assert webhook.call_count == 1
    body = cast("dict[str, Json]", json.loads(webhook.calls.last.request.content))
    (embed,) = cast("list[dict[str, Json]]", body["embeds"])
    return embed


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
    """A title's grab: what it asks of qBittorrent, lists, shows, and caches."""

    def test_a_grab_adds_its_release_lists_and_shows_it_and_caches_the_title(self) -> None:
        recording = install_recording_hub()
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1))}))

        assert (torrents.calls, torrents.dry_runs) == (["h1"], [False])
        # None of the release's files placed, so its row names the title's coverage.
        assert state.stats.grabbed == [
            GrabbedRow(title="Show", coverage=_TITLE_COVERAGE, url=_TITLE_URL, name="Show-RG", group="RG")
        ]
        assert recording.of_type(GrabAction) == [
            GrabAction(
                status=GrabStatus.GRABBING,
                groups=(GroupLabel("RG"),),
                grabbed=(ReleaseLabel("Show-RG", "RG"),),
                in_qbit=(),
                waiting_to_import=True,
            )
        ]
        assert _cached_at(pipeline) == _UPDATED_AT

    def test_the_grabbed_row_names_the_placed_episodes_by_the_numbers_they_print_by(self) -> None:
        # Sonarr lists the aired S02E01 as S01E04.
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), state=state)
        release = _release(1)
        release.episodes = [PlacedEpisode(2, 1, 10, printed=PrintedKey(EpisodeKey(1, 4)))]

        pipeline.grab_and_cache(_grab_request({"RG": _group(release)}))

        assert [row.coverage for row in state.stats.grabbed] == ["S01 E04"]

    def test_a_grab_with_no_fallback_in_it_clears_the_titles_stored_fallback_marker(self) -> None:
        # The cache write merges columns, so a marker an earlier fallback grab left stays unless every write sets it.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}))
        pipeline.cache_store.update_cached_title(Arr.SONARR, 42, {"fallback_satisfied": True})

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1))}))

        cached = pipeline.cache_store.get_cached_title(Arr.SONARR, 42)
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_a_title_with_nothing_to_grab_is_counted_and_shown_up_to_date_and_cached(self) -> None:
        recording = install_recording_hub()
        state = _run_state()
        # Two titles of the run were up to date before this one.
        state.stats.up_to_date = 2
        torrents = FakeTorrents({})
        pipeline = _pipeline(torrents=torrents, state=state)
        # Not marked to grab: Sonarr has the release's files.
        owned = make_release(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash="h1")

        pipeline.grab_and_cache(_grab_request({"RG": _group(owned)}))

        assert torrents.calls == []
        assert state.stats.up_to_date == 3
        assert _statuses(recording) == ["up to date"]
        assert _cached_at(pipeline) == _UPDATED_AT

    @pytest.mark.parametrize(
        ("dry_run", "qbittorrent"),
        [pytest.param(True, True, id="asked-for"), pytest.param(False, False, id="no-qbittorrent")],
    )
    @respx.mock
    def test_a_dry_run_asks_each_add_as_a_dry_run_and_tracks_posts_and_holds_nothing(
        self, dry_run: bool, qbittorrent: bool
    ) -> None:
        # Two releases under a cap of 1: a real run would add the first and hold the second.
        recording = install_recording_hub()
        webhook = respx.post(_DISCORD_URL).respond(json={})
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG 1"), "h2": (AddStatus.ADDED, "Show-RG 2")})
        state = _run_state(dry_run=dry_run)
        pipeline = _pipeline(
            torrents=torrents, state=state, qbittorrent=qbittorrent, max_torrents_to_add=1, _notifier=_discord()
        )
        updates = {"h1": pending_import_update("h1"), "h2": pending_import_update("h2")}

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1), _release(2))}, pending_import_updates=updates))

        assert (torrents.calls, torrents.dry_runs) == (["h1", "h2"], [True, True])
        assert (_tracked(pipeline), _guards(pipeline), state.grabbed_this_run) == (set(), {}, {})
        assert not webhook.called
        assert (state.stats.held_by_cap, recording.of_type(CapReached)) == (0, [])
        assert recording.of_type(GrabAction) == [
            GrabAction(
                status=GrabStatus.WOULD_GRAB,
                groups=(GroupLabel("RG"),),
                grabbed=(ReleaseLabel("Show-RG 1", "RG"), ReleaseLabel("Show-RG 2", "RG")),
                in_qbit=(),
                waiting_to_import=False,
            )
        ]


class TestGrabReleaseRegistersPending:
    """A grabbed release's pending import is stored from its update, and a fresh add is a grab that counts toward the cap."""

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-NAN0")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)
        facts = OwnedAtGrab(entry_groups=("NAN0",), stale_groups=("OldPick",))
        updates = {"h1": pending_import_update("h1", guards=facts)}

        n_added, _ = pipeline.grab_releases(
            _grab_request({"NAN0": _group(_release(1))}, pending_import_updates=updates)
        )

        assert _tracked(pipeline) == {"h1"}
        assert _guards(pipeline) == {PENDING_IMPORT_ANILIST_ID: facts}
        assert [p.infohash for p in state.grabbed_this_run.values()] == ["h1"]
        assert n_added == 1
        assert state.torrents_added == 1

    def test_a_second_pending_import_update_carrying_the_first_pending_import_joins_its_claim(self) -> None:
        # Two titles share one torrent in one run: A's add tracks the pending import fresh, B's dedups to
        # ALREADY_IN_QBIT with an update carrying A's pending import (the store read). B's claim joins it, the
        # run-list copy follows, and a this-run grab is never a reacquire.
        store = FakeCacheStore()
        state = _run_state()
        first = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show")}), state=state, cache_store=store)
        first.grab_releases(
            _grab_request(
                {"NAN0": _group(_release(1))},
                pending_import_updates={"h1": pending_import_update("h1", anilist_id=11, title="Cour 1")},
            )
        )
        second = _pipeline(
            torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), state=state, cache_store=store
        )
        update = pending_import_update("h1", anilist_id=22, title="Cour 2", stored=_stored(first, "h1"))

        second.grab_releases(_grab_request({"NAN0": _group(_release(1))}, pending_import_updates={"h1": update}))

        assert claim_anilist_ids(_stored(second, "h1")) == (11, 22)
        assert claim_anilist_ids(state.grabbed_this_run["h1"]) == (11, 22)
        assert state.reacquired_keys == set()
        assert _guards(second).keys() == {11, 22}

    def test_radarr_registration_writes_no_guard_row(self) -> None:
        # Guard evidence is Sonarr-only enforcement: a Radarr state driven through
        # the same path persists its pending import but never a guard row.
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Movie")})
        pipeline = make_grab_pipeline(
            _torrents=torrents,
            _state=ArrRunState(arr=Arr.RADARR, import_wait_mode=ImportWaitMode.BLOCKING),
        )
        updates = {"h1": pending_import_update("h1", series_id=0)}

        pipeline.grab_releases(_grab_request({"NAN0": _group(_release(1))}, pending_import_updates=updates))

        assert set(pipeline.cache_store.get_pending_import_rows(Arr.RADARR)) == {"h1"}
        assert pipeline.cache_store.get_guards(Arr.RADARR) == {}
        assert pipeline.cache_store.get_guards(Arr.SONARR) == {}


class TestReacquireRegistration:
    """A torrent already in qBittorrent joins the stored pending import its update carries, or starts one.

    A new one starts at qBittorrent's add time. A fresh add of a torrent with a stored pending import restarts it now.
    """

    _STORED_AT = "2026-01-01 00:00:00"
    # Long before the stored start, as a torrent added to qBittorrent by hand months ago.
    _QBITTORRENT_ADDED_ON = datetime(2025, 9, 1, 12, 0, 0)

    def _reacquire(self, added_on: datetime | None) -> FakeTorrents:
        return FakeTorrents({"h1": AddResult(AddStatus.ALREADY_IN_QBIT, "Show", added_on)})

    def _resident(self, pipeline: GrabPipeline) -> PendingImport:
        """A carried-over pending import put in the store, stamped well before this run."""

        resident = pending_import(infohash="h1", added_at=self._STORED_AT)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, resident.infohash, resident.to_json())
        return resident

    def _add(self, pipeline: GrabPipeline, update: PendingImportUpdate) -> list[GrabResult]:
        _, results = pipeline.grab_releases(
            _grab_request({"NAN0": _group(_release(1))}, pending_import_updates={"h1": update})
        )
        return results

    def test_a_joining_title_keeps_the_stored_start(self) -> None:
        # A second title's update carrying the stored pending import joins it. The pending import keeps its own
        # `added_at` whatever qBittorrent's add time says, and a carried-over pending import is a reacquire (never a
        # run-list insert).
        state = _run_state()
        pipeline = _pipeline(torrents=self._reacquire(self._QBITTORRENT_ADDED_ON), state=state)
        resident = self._resident(pipeline)

        self._add(pipeline, pending_import_update("h1", anilist_id=22, stored=resident))

        pending = _stored(pipeline, "h1")
        assert pending.added_at == self._STORED_AT
        assert claim_anilist_ids(pending) == (PENDING_IMPORT_ANILIST_ID, 22)
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
        state = _run_state()
        pipeline = _pipeline(torrents=self._reacquire(self._QBITTORRENT_ADDED_ON), state=state)

        results = self._add(pipeline, pending_import_update("h1"))

        assert _stored(pipeline, "h1").added_at == "2025-09-01 12:00:00"
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
        release = _release(1)
        release.cross_seed_hash = "c1"

        pipeline.grab_releases(
            _grab_request({"NAN0": _group(release)}, pending_import_updates={"c1": pending_import_update("c1")})
        )

        assert _tracked(pipeline) == {"c1"}

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
        assert claim_anilist_ids(pending) == (PENDING_IMPORT_ANILIST_ID, 22)
        assert before <= pending.added_at <= now_stamp()
        assert state.reacquired_keys == set()
        assert list(state.grabbed_this_run) == ["h1"]


class TestTheRunCap:
    """A title's grabbable releases past the run's cap of adds are held, and the title says so and stays uncached.

    The title whose adds reach the cap announces it, once. A run with no cap in force holds none and announces none.
    """

    @pytest.mark.parametrize(
        ("already_added", "asked", "tracked"),
        [
            pytest.param(1, [], set[str](), id="past-the-cap"),
            pytest.param(0, ["h1"], {"h1"}, id="crosses-the-cap-mid-title"),
        ],
    )
    def test_a_release_past_the_cap_holds_its_title_once_and_leaves_it_uncached(
        self, already_added: int, asked: list[str], tracked: set[str]
    ) -> None:
        # Two releases to grab under a cap of 1. A held release new to qBittorrent is not tracked yet.
        recording = install_recording_hub()
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG 1"), "h2": (AddStatus.ADDED, "Show-RG 2")})
        state = _run_state(torrents_added=already_added)
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)
        updates = {"h1": pending_import_update("h1"), "h2": pending_import_update("h2")}

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1), _release(2))}, pending_import_updates=updates))

        assert torrents.calls == asked
        assert _tracked(pipeline) == tracked
        assert _statuses(recording) == ["held by the run cap; grabbed next run"]
        assert (state.stats.held_by_cap, state.stats.needs_action, _cached_at(pipeline)) == (1, [], None)

    @pytest.mark.parametrize(
        ("already_added", "announced", "cached"),
        [
            # Reaching the cap holds nothing of this title, so it caches.
            pytest.param(0, [1], _UPDATED_AT, id="reaches-the-cap"),
            pytest.param(1, [], None, id="already-past-it"),
        ],
    )
    def test_only_the_title_whose_add_reaches_the_cap_announces_it(
        self, already_added: int, announced: list[int], cached: str | None
    ) -> None:
        recording = install_recording_hub()
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, state=_run_state(torrents_added=already_added), max_torrents_to_add=1)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1))}))

        assert ([e.cap for e in recording.of_type(CapReached)], _cached_at(pipeline)) == (announced, cached)

    def test_a_joining_title_held_by_the_cap_saves_its_claim(self) -> None:
        # A torrent already in qBittorrent under a stored pending import stays tracked past the cap: the title's claim
        # joins the pending import without a qBittorrent call, and nothing joins the reacquired keys.
        torrents = FakeTorrents({})
        state = _run_state(torrents_added=1)
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)
        resident = pending_import(infohash="h1", anilist_id=11)
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", anilist_id=22, stored=resident)}

        pipeline.grab_releases(_grab_request({"RG": _group(_release(1))}, pending_import_updates=updates))

        assert torrents.calls == []
        assert claim_anilist_ids(_stored(pipeline, "h1")) == (11, 22)
        assert state.reacquired_keys == set()
        assert state.grabbed_this_run == {}

    def test_a_torrent_already_in_qbittorrent_is_no_grab_and_the_release_after_it_still_adds(self) -> None:
        # Under a cap of 1, a torrent already there that counted toward it would hold the release after it.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "old"), "h2": (AddStatus.ADDED, "new")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state, max_torrents_to_add=1)

        n_added, results = pipeline.grab_releases(_grab_request({"RG": _group(_release(1), _release(2))}))

        assert torrents.calls == ["h1", "h2"]
        assert (n_added, state.torrents_added) == (1, 1)
        assert [r.outcome for r in results] == [AddStatus.ALREADY_IN_QBIT, AddStatus.ADDED]
        assert [row.name for row in state.stats.grabbed] == ["new"]

    def test_a_cap_of_zero_adds_every_release_and_announces_no_cap(self) -> None:
        recording = install_recording_hub()
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "one"), "h2": (AddStatus.ADDED, "two")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=0)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1), _release(2))}))

        assert torrents.calls == ["h1", "h2"]
        assert recording.of_type(CapReached) == []

    def test_a_release_screened_out_past_the_cap_posts_its_skip_and_holds_nothing(self) -> None:
        recording = install_recording_hub()
        state = _run_state(torrents_added=1)
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state, max_torrents_to_add=1)

        pipeline.grab_releases(_grab_request({"RG": _group(_release(1, skipped=SkipReason.MISNUMBERED))}))

        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == [SkipReason.MISNUMBERED]
        assert state.stats.held_by_cap == 0


class TestTrackerSkips:
    """A release on a private tracker or one with no parser is never added.

    A title that grabbed nothing else stays uncached under one row saying why. One that grabbed something beside it
    is cached, with no row.
    """

    @pytest.mark.parametrize(
        ("skipped", "private_releases", "interactive", "reason", "cause"),
        [
            pytest.param(
                [SkipReason.UNSUPPORTED_TRACKER],
                PrivateReleaseAction.WARN,
                False,
                "tracker not yet supported; grab manually",
                NeedsActionCause.UNSUPPORTED_TRACKER,
                id="no-parser",
            ),
            # One group with a public release keeps its private one marked to grab. The two reasons are either/or:
            # private-only wins.
            pytest.param(
                [SkipReason.PRIVATE_ONLY, SkipReason.UNSUPPORTED_TRACKER],
                PrivateReleaseAction.WARN,
                False,
                "private-only release; private releases not supported",
                NeedsActionCause.PRIVATE_ONLY,
                id="private-beside-no-parser",
            ),
            # Interactive and fallback: the private release was picked by hand, and the cause stays the one whose
            # tip never suggests the fallback.
            pytest.param(
                [SkipReason.PRIVATE_ONLY],
                PrivateReleaseAction.FALLBACK,
                True,
                "hand-picked private release; private releases not supported",
                NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
                id="hand-picked-private",
            ),
        ],
    )
    def test_a_title_with_nothing_grabbable_is_left_uncached_under_one_row_saying_why(
        self,
        skipped: list[SkipReason],
        private_releases: PrivateReleaseAction,
        interactive: bool,
        reason: str,
        cause: NeedsActionCause,
    ) -> None:
        recording = install_recording_hub()
        releases = [_release(n, skipped=skip) for n, skip in enumerate(skipped, start=1)]
        torrents = FakeTorrents({})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state, private_releases=private_releases, interactive=interactive)

        pipeline.grab_and_cache(_grab_request({"NAN0": _group(*releases)}))

        assert [e.reason for e in recording.of_type(ReleaseSkipped)] == skipped
        assert torrents.calls == []
        assert state.stats.needs_action == [_needs_action("NAN0", reason, cause)]
        assert _cached_at(pipeline) is None

    def test_a_title_whose_private_only_release_the_plan_dropped_is_neither_up_to_date_nor_cached(self) -> None:
        # Not interactive, the plan drops a lone private group's grab and flags the title itself, so the pipeline
        # meets a title with nothing to grab.
        recording = install_recording_hub()
        state = _run_state()
        private = make_release(
            url="https://animebytes.tv/torrents.php?id=1",
            files=["Show - 01.mkv"],
            size=[100],
            tracker=Tracker.ANIMEBYTES,
            is_public=False,
            infohash=None,
        )
        in_sonarr = [sonarr_ep(1, 1, episode_file_id=0)]
        plan = make_release_filter(state=state).plan_grabs(
            {"RG": rg_group({private.url: private}, episodes=[placed_episode(1, 1, 100)])},
            arr_files(in_sonarr),
            in_sonarr,
        )
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state, private_releases=PrivateReleaseAction.FALLBACK)

        pipeline.grab_and_cache(_grab_request(plan.seadex_dict))

        assert private.to_grab is False
        assert (state.stats.up_to_date, _statuses(recording), _cached_at(pipeline)) == (0, [], None)
        assert state.stats.needs_action == [
            _needs_action(
                "RG", "private-only release; no fallback covers these files", NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK
            )
        ]

    @pytest.mark.parametrize(
        ("skipped", "private_releases", "interactive"),
        [
            pytest.param(SkipReason.UNSUPPORTED_TRACKER, PrivateReleaseAction.WARN, False, id="no-parser"),
            pytest.param(SkipReason.PRIVATE_ONLY, PrivateReleaseAction.FALLBACK, True, id="hand-picked-private"),
        ],
    )
    def test_a_grab_beside_a_tracker_skip_caches_its_title_with_no_row(
        self, skipped: SkipReason, private_releases: PrivateReleaseAction, interactive: bool
    ) -> None:
        state = _run_state()
        pipeline = _pipeline(
            torrents=FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}),
            state=state,
            private_releases=private_releases,
            interactive=interactive,
        )

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1, skipped=skipped), _release(2))}))

        assert (state.torrents_added, state.stats.needs_action, _cached_at(pipeline)) == (1, [], _UPDATED_AT)


class TestPlacementSkips:
    """A release the placement skipped is never grabbed: the title stays uncached, and the summary says why.

    One skipped over episodes Sonarr has no row for is never grabbed either, and its title is cached.
    """

    @pytest.mark.parametrize(
        ("skipped", "counts", "reason", "cause"),
        [
            pytest.param(
                SkipReason.MISNUMBERED,
                NO_ANOTHER_TITLES_COUNTS,
                "numbering does not match its SeaDex listing; import it by hand",
                NeedsActionCause.MISNUMBERED,
                id="misnumbered",
            ),
            pytest.param(
                SkipReason.EPISODE_TITLES_CONTRADICT,
                NO_ANOTHER_TITLES_COUNTS,
                "episode titles contradict its numbers; import it by hand",
                NeedsActionCause.EPISODE_TITLES_CONTRADICT,
                id="titles-refute",
            ),
            pytest.param(
                SkipReason.PLACED_ELSEWHERE,
                AnotherTitlesCounts(another_titles=12, unplaced=6),
                "its files place under another title SeaDex lists it with; any it could not place need a hand import",
                NeedsActionCause.PLACED_ELSEWHERE,
                id="another-titles-files",
            ),
            # A read the placement needs failed while the release still had a file to place.
            pytest.param(
                SkipReason.PLACEMENT_INPUT_UNREAD,
                NO_ANOTHER_TITLES_COUNTS,
                "a read the placement needs failed; will retry next run",
                NeedsActionCause.PLACEMENT_INPUT_UNREAD,
                id="a-read-failed",
            ),
        ],
    )
    def test_a_skipped_release_is_never_added_and_its_skip_and_row_say_why(
        self, skipped: SkipReason, counts: AnotherTitlesCounts, reason: str, cause: NeedsActionCause
    ) -> None:
        recording = install_recording_hub()
        release = _release(1, skipped=skipped)
        release.another_titles_counts = counts
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(release)}))

        assert torrents.calls == []
        assert _cached_at(pipeline) is None
        assert [(e.reason, e.counts) for e in recording.of_type(ReleaseSkipped)] == [(skipped, counts)]
        assert state.stats.needs_action == [_needs_action("RG", reason, cause)]

    def test_a_grab_beside_a_placement_skip_leaves_its_title_uncached(self) -> None:
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}), state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1, skipped=SkipReason.MISNUMBERED), _release(2))}))

        assert state.torrents_added == 1
        assert [row.cause for row in state.stats.needs_action] == [NeedsActionCause.MISNUMBERED]
        assert _cached_at(pipeline) is None

    def test_a_release_on_episodes_sonarr_has_no_row_for_is_planned_but_never_added_and_its_title_is_cached(
        self,
    ) -> None:
        # The planner marks the release to grab by its group and sizes: none of its files placed, and the title's
        # one episode in Sonarr has no file.
        recording = install_recording_hub()
        nyaa = make_release(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash="h1", size=[100, 200])
        nyaa.skip_reason = SkipReason.AIRED_ONLY
        in_sonarr = [sonarr_ep(1, 3, episode_file_id=0)]
        plan = make_planner().plan({"RG": rg_group({nyaa.url: nyaa}, episodes=[])}, arr_files(in_sonarr), in_sonarr)
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request(plan.seadex_dict))

        assert nyaa.to_grab is True
        assert torrents.calls == []
        assert _cached_at(pipeline) == _UPDATED_AT
        assert [(e.reason, e.severity) for e in recording.of_type(ReleaseSkipped)] == [
            (SkipReason.AIRED_ONLY, Severity.INFO)
        ]
        assert state.stats.needs_action == []

    def test_a_skipped_release_of_a_resident_torrent_never_joins_its_pending_import(self) -> None:
        # Another title's pending import has the torrent.
        resident = pending_import(infohash="h1")
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}))
        pipeline.cache_store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        updates = {"h1": pending_import_update("h1", anilist_id=22, stored=resident)}

        pipeline.grab_releases(
            _grab_request(
                {"RG": _group(_release(1, skipped=SkipReason.EPISODE_TITLES_CONTRADICT))},
                pending_import_updates=updates,
            )
        )

        assert claim_anilist_ids(_stored(pipeline, "h1")) == (PENDING_IMPORT_ANILIST_ID,)

    def test_a_release_not_to_grab_with_a_skip_reason_is_not_skipped_and_the_title_caches(self) -> None:
        # The arr has the torrent's files now (the hand import is done), and another release of the title still grabs.
        recording = install_recording_hub()
        done = _release(1, skipped=SkipReason.MISNUMBERED)
        done.to_grab = False
        pipeline = _pipeline(torrents=FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}))

        pipeline.grab_and_cache(_grab_request({"RG": _group(done, _release(2))}))

        assert recording.of_type(ReleaseSkipped) == []
        assert _cached_at(pipeline) == _UPDATED_AT


class TestPlacementInputUnread:
    """A failed Sonarr read under a grab-time placement leaves the title uncached, with a retry row in the summary."""

    def test_a_grabbed_title_stays_uncached_with_a_retry_row(self) -> None:
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddStatus.ADDED, "Show-RG")}), state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1))}, input_unread_groups=("RG",)))

        assert [row.name for row in state.stats.grabbed] == ["Show-RG"]
        assert _cached_at(pipeline) is None
        assert state.stats.needs_action == [
            _needs_action(
                "RG",
                "a read the placement needs failed; will retry next run",
                NeedsActionCause.PLACEMENT_INPUT_UNREAD,
            )
        ]

    def test_a_title_with_nothing_to_grab_is_neither_up_to_date_nor_cached(self) -> None:
        # The placement may have left a sequence waiting, so the coverage decision was coarse: no "already have it".
        owned = make_release(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash="h1")
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({}), state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(owned)}, input_unread_groups=("RG",)))

        assert state.stats.up_to_date == 0
        assert _cached_at(pipeline) is None
        assert [r.cause for r in state.stats.needs_action] == [NeedsActionCause.PLACEMENT_INPUT_UNREAD]


def test_a_title_read_without_its_metadata_stays_uncached_with_nothing_to_do() -> None:
    # AniList or SkyHook was unreachable before the first fetch: the title re-checks next run, and no row asks for a
    # hand.
    owned = make_release(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash="h1")
    state = _run_state()
    pipeline = _pipeline(torrents=FakeTorrents({}), state=state)

    pipeline.grab_and_cache(replace(_grab_request({"RG": _group(owned)}), metadata_unread=True))

    assert _cached_at(pipeline) is None
    assert state.stats.needs_action == []


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent unreachable or erroring) is contained at the add.

    One warning, no traceback: the release loop moves on, the title stays uncached (retried next run), and the summary
    carries a GRAB_FAILED needs-action row instead of the title silently vanishing. Any other error is a bug, and gets
    out.
    """

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(TrackerPageError("Could not find the torrent title on https://nyaa.si/view/1"), id="parse"),
            pytest.param(TorrentAddError("qBittorrent rejected the torrent"), id="add"),
            pytest.param(httpx.ConnectError("tracker down"), id="tracker"),
            pytest.param(qbittorrentapi.APIConnectionError("qbit died mid-run"), id="qbit"),
        ],
    )
    def test_failure_is_one_clean_warning_no_traceback(self, error: Exception) -> None:
        # Each kind of error a tracker or qBittorrent raises is caught at the add, and posted as one warning naming
        # the release.
        torrents = FakeTorrents({}, raises={"h1": error})
        pipeline = _pipeline(torrents=torrents)
        recording = install_recording_hub()

        n_added, results = pipeline.grab_releases(_grab_request({"NAN0": _group(_release(1))}))

        assert (n_added, results) == (0, [])
        (failed,) = recording.of_type(GrabFailed)
        assert failed == GrabFailed(group="NAN0", url="https://nyaa.si/view/1", error=str(error))
        assert severity_of(failed) is Severity.WARNING

    def test_a_failed_release_never_drops_the_next_and_leaves_the_title_uncached(self) -> None:
        # Containment is per release: the sibling after the failure still grabs, and the title retries next run.
        torrents = FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}, raises={"h1": httpx.ConnectError("nyaa down")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1), _release(2))}))

        assert torrents.calls == ["h1", "h2"]
        assert [row.name for row in state.stats.grabbed] == ["Show-RG"]
        assert [row.cause for row in state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]
        assert _cached_at(pipeline) is None

    def test_failed_only_title_stays_uncached_with_a_retry_row(self) -> None:
        # Nyaa is down for both of the group's torrents, and the row names the group once.
        down = httpx.ConnectError("nyaa down")
        state = _run_state()
        pipeline = _pipeline(torrents=FakeTorrents({}, raises={"h1": down, "h2": down}), state=state)

        pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1), _release(2))}))

        assert state.stats.grabbed == []
        assert _cached_at(pipeline) is None
        assert state.stats.needs_action == [
            _needs_action("RG", "grab failed; will retry next run", NeedsActionCause.GRAB_FAILED)
        ]

    def test_an_error_that_is_no_tracker_or_qbittorrent_failure_gets_out(self) -> None:
        # Contained, a bug in the add would read as a failed grab, retried every run with no traceback.
        pipeline = _pipeline(torrents=FakeTorrents({}, raises={"h1": RuntimeError("a bug in the add")}))

        with pytest.raises(RuntimeError, match="a bug in the add"):
            pipeline.grab_and_cache(_grab_request({"RG": _group(_release(1))}))

    def test_a_failed_grab_on_one_title_never_keeps_the_next_title_uncached(self) -> None:
        # Two titles in one run, each through the real per-title head: the first title's add fails, the second's
        # goes through.
        torrents = FakeTorrents({"h2": (AddStatus.ADDED, "Show-RG")}, raises={"h1": httpx.ConnectError("nyaa down")})
        state = _run_state()
        pipeline = _pipeline(torrents=torrents, state=state)
        entries = {1: make_entry_record(anilist_id=1), 2: make_entry_record(anilist_id=2)}
        services = make_services(_seadex=FakeSeaDexSource(entries), _state=state, _grab_pipeline=pipeline)

        for anilist_id, release in ((1, _release(1)), (2, _release(2))):
            entry = services.begin_check(anilist_id)
            assert entry is not None
            services.grab_and_cache(
                grab_request(
                    anilist_id=anilist_id,
                    entry=entry,
                    seadex_dict={"RG": _group(release)},
                    update={"updated_at": _UPDATED_AT},
                )
            )

        assert [row.cause for row in state.stats.needs_action] == [NeedsActionCause.GRAB_FAILED]
        assert (_cached_at(pipeline, 1), _cached_at(pipeline, 2)) == (None, _UPDATED_AT)


@pytest.mark.parametrize(("wins", "loses"), list(combinations(_ROW_PRECEDENCE, 2)), ids=lambda row: row[1].value)
def test_two_reasons_on_one_title_list_it_once_under_the_one_that_outranks_the_other(
    wins: tuple[SkipReason | None, NeedsActionCause], loses: tuple[SkipReason | None, NeedsActionCause]
) -> None:
    # The losing release comes first, so the order they are met in decides nothing. Interactive, so a private
    # release reaches the add loop as a hand pick.
    first, second = _release(1, skipped=loses[0]), _release(2, skipped=wins[0])
    down = httpx.ConnectError("nyaa down")
    state = _run_state()
    pipeline = _pipeline(torrents=FakeTorrents({}, raises={"h1": down, "h2": down}), state=state, interactive=True)

    pipeline.grab_and_cache(_grab_request({"A": _group(first), "B": _group(second)}))

    assert [row.cause for row in state.stats.needs_action] == [wins[1]]


class TestGrabPushesNotice:
    """A title's grab posts one Discord notice built from its request and what the adds did, never when nothing was added.

    A dry run posts none (`TestAGrab`).
    """

    @respx.mock
    def test_added_pushes_the_title_its_art_its_coverage_and_what_it_replaces(self) -> None:
        webhook = respx.post(_DISCORD_URL).respond(json={})
        pipeline = _notifying_pipeline(FakeTorrents({"h1": (AddStatus.ADDED, "Show-PMR")}))

        pipeline.grab_and_cache(
            GrabRequest(
                anilist_id=42,
                # The library item's name must not prefix the AniList one, or the byline would (rightly) dedupe away.
                arr_title="The Show",
                title_name="Show Title",
                entry=make_entry_record(anilist_id=42, url="https://releases.moe/42", notes="the why"),
                seadex_dict={"PMR": _group(_release(1))},
                update={},
                replaced_groups=("OldGroup",),
                coverage="S01 E01-E12",
            )
        )

        embed = _posted_embed(webhook)
        assert cast("dict[str, Json]", embed["author"])["name"] == "Sonarr · SeaDex grab"
        assert (embed["title"], embed["url"]) == ("Show Title", "https://releases.moe/42")
        assert (embed["thumbnail"], embed["image"]) == ({"url": "https://img/large"}, {"url": "https://img/banner"})
        # A single-group grab hoists its release into the description. The byline and notes trail as the nameless
        # field.
        assert embed["description"] == "**Grabbed · `PMR`**\n[Nyaa](https://nyaa.si/view/1)"
        assert embed["fields"] == [
            {"name": "Episodes", "value": "S01 E01-E12", "inline": True},
            {"name": "Replacing", "value": "OldGroup", "inline": True},
            {"name": "", "value": "-# The Show\n> the why", "inline": False},
        ]

    @respx.mock
    def test_a_group_whose_add_failed_is_named_failed_beside_the_grabbed_one(self) -> None:
        # Nyaa was down for group Bad's release. Its block must not read as skipped, which would say it was never tried.
        webhook = respx.post(_DISCORD_URL).respond(json={})
        torrents = FakeTorrents({"h1": (AddStatus.ADDED, "Show-PMR")}, raises={"h2": httpx.ConnectError("nyaa down")})
        pipeline = _notifying_pipeline(torrents)

        pipeline.grab_and_cache(_grab_request({"PMR": _group(_release(1)), "Bad": _group(_release(2))}))

        embed = _posted_embed(webhook)
        assert embed["fields"] == [
            {"name": "Grabbed · PMR", "value": "[Nyaa](https://nyaa.si/view/1)", "inline": False},
            {"name": "Failed · Bad", "value": "[Nyaa](https://nyaa.si/view/2)", "inline": False},
        ]

    @respx.mock
    def test_nothing_added_pushes_nothing(self) -> None:
        webhook = respx.post(_DISCORD_URL).respond(json={})
        pipeline = _notifying_pipeline(FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show-PMR")}))

        pipeline.grab_and_cache(_grab_request({"PMR": _group(_release(1))}))

        assert not webhook.called
