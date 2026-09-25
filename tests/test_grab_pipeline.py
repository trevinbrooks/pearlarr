# pyright: strict
# pyright: reportPrivateUsage=false
# The add-path assertions read the pipeline's private wiring (_grab / _ctx), which
# strict re-flags. The repo disables reportPrivateUsage for tests.
"""Unit tests for the grab "produce" side (`GrabPipeline`).

Pin the add path - `_add_one_url` registering durable `PendingImport`
records, `add_torrent`'s cap bookkeeping, and `_grab` returning a pure
cap-reached bool (it never finalizes. The engine owns the single finalize site).
Built bare (`object.__new__` via `make_bare_instance`) so no live qBittorrent
login happens. The client `add` is faked by `FakeTorrents`.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
import qbittorrentapi
from seadex import Tracker

from pearlarr import notify
from pearlarr.config import Arr
from pearlarr.discord import DiscordEmbed
from pearlarr.grab_pipeline import NO_SEEDS, GrabPipeline, GrabRequest
from pearlarr.grab_placement import PendingSeed
from pearlarr.manual_import import GuardFacts, ImportWaitMode, PendingImport, normalize_basename
from pearlarr.notify import Notifier
from pearlarr.output import CapReached, EntryDetail, GrabFailed, Severity, install_hub, severity_of
from pearlarr.output.recording import RecordingHub
from pearlarr.reporter import NeedsActionKind, PerTitleState, RunContext
from pearlarr.seadex_types import SeadexDict, SeadexUrlItem
from pearlarr.stamps import now_stamp, stamp_of
from pearlarr.torrent import TorrentParseError
from pearlarr.torrents import AddResult, ReleaseOutcome, TorrentAddError

from .builders import (
    CLIENT_SENTINEL,
    PENDING_AL_ID,
    AddOutcome,
    FakeCacheStore,
    FakeTorrents,
    claim_al_ids,
    grab_request,
    make_entry_record,
    make_grab_pipeline,
    one_release_dict,
    pending_import,
    pending_seed,
    rg_group,
    url_item,
)
from .fakes import FakeClock, install_recording_hub


def _stub_add_torrent(req: GrabRequest) -> tuple[int, list[ReleaseOutcome]]:
    """Replaces `GrabPipeline.add_torrent` for the cap-notice tests.

    Returns a fixed `(n_added, results)` so `_grab`'s cap notice is exercised without a real qBittorrent add. The
    counter it reads (`ctx.torrents_added`) is set by the test.
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
    """A bare `GrabPipeline` wired for the add path (a non-preview blocking run)."""

    return make_grab_pipeline(
        _torrents=torrents,
        qbit=qbit,
        _ctx=RunContext(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode),
        **config,
    )


def _pending(pipeline: GrabPipeline) -> Mapping[str, dict[str, Any]]:
    """The pipeline's durable per-arr pending store, keyed by infohash (what the engine reads back)."""

    return pipeline.cache_store.get_pending(Arr.SONARR)


def _stored(pipeline: GrabPipeline, infohash: str) -> PendingImport:
    """The store's record on the torrent, rehydrated (its stamps and claims need no guard row)."""

    raw = pipeline.cache_store.get_pending_record(Arr.SONARR, infohash)
    assert raw is not None
    return PendingImport.from_json(raw, guards={})


def _guards(pipeline: GrabPipeline) -> Mapping[int, GuardFacts]:
    """The pipeline's durable per-entry guard rows (what the read seams hydrate from)."""

    return pipeline.cache_store.get_guards(Arr.SONARR)


class TestGrabAnnouncesTheCap:
    """`_grab` returns the added count and posts the cap notice once, from the title whose adds crossed it.

    GrabPipeline holds no reference back to the engine, so the notice is the only cap signal it emits: the scan
    never stops on it.
    """

    @staticmethod
    def _request() -> GrabRequest:
        return grab_request(entry=make_entry_record(url="https://seadex.example/1"))

    def test_the_title_crossing_the_cap_announces_it(self) -> None:
        # The stub reports one add without touching the counter, so the counter is advanced by hand to the cap.
        recording = install_recording_hub()
        pipeline = make_grab_pipeline(max_torrents_to_add=1, add_torrent=_stub_add_torrent)
        # Warm the gateway cache so the embed's thumb lookup never hits AniList.
        pipeline._anilist.al_cache.update({1: {}})
        pipeline._ctx.torrents_added = 0

        def add_and_count(req: GrabRequest) -> tuple[int, list[ReleaseOutcome]]:
            pipeline._ctx.torrents_added += 1
            return _stub_add_torrent(req)

        pipeline.add_torrent = add_and_count

        assert pipeline._grab(self._request()) == 1
        assert [e.cap for e in recording.of_type(CapReached)] == [1]

    def test_a_title_already_past_the_cap_stays_quiet(self) -> None:
        # The notice belongs to the crossing title alone: a later title adds nothing and re-announces nothing.
        recording = install_recording_hub()
        pipeline = make_grab_pipeline(max_torrents_to_add=1, add_torrent=_stub_add_torrent)
        pipeline._anilist.al_cache.update({1: {}})
        pipeline._ctx.torrents_added = 1  # already at the cap of 1

        pipeline._grab(self._request())

        assert recording.of_type(CapReached) == []

    def test_a_preview_never_announces_the_cap(self) -> None:
        # MUTATION PIN: reading the raw config cap instead of _effective_cap would announce a cap a preview
        # never enforces.
        recording = install_recording_hub()
        pipeline = make_grab_pipeline(qbit=None, max_torrents_to_add=1, add_torrent=_stub_add_torrent)
        pipeline._anilist.al_cache.update({1: {}})
        pipeline._ctx.torrents_added = 0

        def add_and_count(req: GrabRequest) -> tuple[int, list[ReleaseOutcome]]:
            pipeline._ctx.torrents_added += 1
            return _stub_add_torrent(req)

        pipeline.add_torrent = add_and_count

        pipeline._grab(self._request())

        assert recording.of_type(CapReached) == []


class TestGrabPushesNotice:
    """`_grab` builds a `GrabNotice` from the request and add outcomes, then pushes it.

    It never pushes on a preview run, and never when nothing was actually added.
    """

    def _grab(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        outcome: AddOutcome = AddOutcome.ADDED,
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
                # The arr title must not prefix the AniList one, or the byline
                # this class asserts on would (correctly) dedupe away.
                arr_title="The Show",
                entry_title="Show Title",
                entry=make_entry_record(url="https://releases.moe/7", notes="the why"),
                seadex_dict=one_release_dict(srg="PMR", infohash="h1"),
                torrent_hashes=["h1"],
                cache_details={},
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
        # A single-group grab hoists its pick into the description. The
        # subtitle/notes stack trails as the nameless (header-free) field.
        assert embed.description == "**Grabbed · `PMR`**\n[Nyaa](https://nyaa.si/view/1)"
        assert [f.name for f in embed.fields] == ["Episodes", "Replacing", ""]
        assert embed.fields[-1].value == "-# The Show\n> the why"

    def test_nothing_added_pushes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._grab(monkeypatch, outcome=AddOutcome.ALREADY_ADDED) == []

    def test_preview_never_pushes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A preview simulates the add (so the count is non-zero) but the push is
        # an outward notification and must stay silent.
        assert self._grab(monkeypatch, qbit=None) == []


class TestAddOneUrlRegistersPending:
    """`_add_one_url` persists a seed's record for both a fresh and an already-present torrent.

    It records a grab and counts toward the cap only for a fresh add.
    """

    def test_already_added_registers_pending_import(self) -> None:
        # The recommended release is already in qBittorrent with no record (grabbed
        # before the store held it, or its row dropped): track it for the monitor,
        # but never count it as a this-run grab.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        facts = GuardFacts(entry_groups=("NAN0",))
        seeds = {"h1": pending_seed("h1", guards=facts)}

        n_added, results = pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds=seeds),
        )

        assert claim_al_ids(_stored(pipeline, "h1")) == (PENDING_AL_ID,)
        # A registration writes the entry's guard row too: evidence follows the
        # newest plan, never a frozen per-record copy.
        assert _guards(pipeline) == {PENDING_AL_ID: facts}
        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert pipeline._ctx.pending_imports == {}
        assert n_added == 0
        assert pipeline._ctx.torrents_added == 0
        assert pipeline._ctx.stats.added == []
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        facts = GuardFacts(entry_groups=("NAN0",), stale_groups=("OldPick",))
        seeds = {"h1": pending_seed("h1", guards=facts)}

        n_added, _ = pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds=seeds),
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert _guards(pipeline) == {PENDING_AL_ID: facts}
        assert [p.infohash for p in pipeline._ctx.pending_imports.values()] == ["h1"]
        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1
        assert len(pipeline._ctx.stats.added) == 1

    def test_already_added_does_not_count_toward_cap(self) -> None:
        # One already-present + one fresh, cap 1: only the fresh add counts.
        torrents = FakeTorrents(
            {
                "already": (AddOutcome.ALREADY_ADDED, "old"),
                "fresh": (AddOutcome.ADDED, "new"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        seadex_dict = {
            **one_release_dict(srg="OLD", infohash="already", url="https://nyaa.si/view/1"),
            **one_release_dict(srg="NEW", infohash="fresh", url="https://nyaa.si/view/2"),
        }
        seeds = {"already": pending_seed("already"), "fresh": pending_seed("fresh")}

        n_added, _ = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict, pending_seeds=seeds))

        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1

    def test_a_born_seed_on_a_resident_hash_replaces_the_stored_record(self) -> None:
        # The seed's own `stored` decides the arm, never the store: a second entry's seed built
        # without the record reacquires and overwrites it (one record per torrent, the last writer's).
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show")})
        pipeline = _pipeline(torrents=torrents)
        first = pending_seed("h1", al_id=11, title="Cour 1")
        second = pending_seed("h1", al_id=22, title="Cour 2")

        pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds={"h1": first})
        )
        pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds={"h1": second})
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert claim_al_ids(_stored(pipeline, "h1")) == (22,)

    def test_a_second_seed_carrying_the_first_record_accretes_its_claim(self) -> None:
        # Two entries share one torrent in one run: A's add tracks the record fresh, B's dedups to
        # ALREADY_ADDED with a seed carrying A's record (the store read). B's claim joins it, the
        # run-list copy follows, and a this-run grab is never a reacquire.
        store = FakeCacheStore()
        ctx = RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)
        first = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "Show")}), cache_store=store, _ctx=ctx
        )
        first.add_torrent(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_seeds={"h1": pending_seed("h1", al_id=11, title="Cour 1")},
            )
        )
        second = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show")}), cache_store=store, _ctx=ctx
        )
        seed = pending_seed("h1", al_id=22, title="Cour 2", stored=_stored(first, "h1"))

        second.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds={"h1": seed})
        )

        assert claim_al_ids(_stored(second, "h1")) == (11, 22)
        assert claim_al_ids(ctx.pending_imports["h1"]) == (11, 22)
        assert ctx.reacquired_keys == set()
        assert _guards(second).keys() == {11, 22}

    def test_no_seed_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents)

        pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds={}),
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == {}

    def test_off_mode_registers_nothing_off_the_strategys_empty_seeds(self) -> None:
        # The wait-mode gate is the strategy's: it seeds nothing when the mode is off, and the
        # pipeline persists only what it is fed.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, mode=ImportWaitMode.OFF)

        pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds=NO_SEEDS),
        )

        assert _pending(pipeline) == {}
        assert _guards(pipeline) == {}
        assert pipeline._ctx.pending_imports == {}

    def test_preview_does_not_register_but_returns_outcome(self) -> None:
        # No client -> preview: nothing persisted, but the outcome still surfaces.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, qbit=None)
        seeds = {"h1": pending_seed("h1")}

        _, results = pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds=seeds),
        )

        assert _pending(pipeline) == {}
        assert _guards(pipeline) == {}
        assert pipeline._ctx.pending_imports == {}
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]

    def test_radarr_registration_writes_no_guard_row(self) -> None:
        # Guard evidence is Sonarr-only enforcement: a Radarr ctx driven through
        # the same seam persists its pending record but never a guard row.
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Movie")})
        pipeline = make_grab_pipeline(
            _torrents=torrents,
            _ctx=RunContext(arr=Arr.RADARR, import_wait_mode=ImportWaitMode.BLOCKING),
        )
        seeds = {"h1": pending_seed("h1", series_id=0)}

        pipeline.add_torrent(grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds=seeds))

        assert set(pipeline.cache_store.get_pending(Arr.RADARR)) == {"h1"}
        assert pipeline.cache_store.get_guards(Arr.RADARR) == {}
        assert pipeline.cache_store.get_guards(Arr.SONARR) == {}


class TestReacquireRegistration:
    """`_register_pending_import`'s three arms: a fresh add tracks, an ALREADY_ADDED accretes or reacquires.

    The seed's `stored` decides between the last two, never the store. Only a born
    seed's reacquire is stamped at qBittorrent's add time and gated by
    `imports.pending_max_age_days` (default 14 days).
    """

    _STORED_AT = "2026-01-01 00:00:00"

    def _reacquire(self, added_on: datetime | None) -> FakeTorrents:
        return FakeTorrents({"h1": AddResult(AddOutcome.ALREADY_ADDED, "Show", added_on)})

    def _resident(self, pipeline: GrabPipeline) -> PendingImport:
        """A carried-over record put in the store, stamped well before this run."""

        resident = pending_import(infohash="h1", added_at=self._STORED_AT)
        pipeline.cache_store.put_pending(Arr.SONARR, resident.infohash, resident.to_json())
        return resident

    def _add(self, pipeline: GrabPipeline, seed: PendingSeed) -> list[ReleaseOutcome]:
        _, results = pipeline.add_torrent(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_seeds={"h1": seed}),
        )
        return results

    def test_accretion_keeps_the_stored_clocks(self) -> None:
        # A second entry's seed carrying the stored record joins it: the birth and
        # the first claim keep their stamps, only the new claim is stamped now, and
        # a carried-over record is a reacquire (never a run-list insert).
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        before = now_stamp()

        self._add(pipeline, pending_seed("h1", al_id=22, stored=resident))

        record = _stored(pipeline, "h1")
        assert record.added_at == self._STORED_AT
        assert [c.al_id for c in record.claims] == [PENDING_AL_ID, 22]
        assert record.claims[0].claimed_at == self._STORED_AT
        assert before <= record.claims[1].claimed_at <= now_stamp()
        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert pipeline._ctx.pending_imports == {}

    def test_this_runs_own_grab_is_never_a_reacquire(self) -> None:
        # One entry listing a torrent twice: the second add meets this run's fresh record, and the
        # torrent stays the run's grab.
        pipeline = _pipeline(torrents=self._reacquire(None))
        pipeline._ctx.pending_imports["h1"] = pending_import(infohash="h1")

        self._add(pipeline, pending_seed("h1"))

        assert pipeline._ctx.reacquired_keys == set()

    def test_accretion_never_reads_the_add_time(self) -> None:
        # qBittorrent's add time past the cutoff drops only a born seed: a stored
        # record accretes regardless, prune_expired_pending staying the sole TTL
        # authority over it.
        pipeline = _pipeline(torrents=self._reacquire(datetime.now() - timedelta(days=100)))
        resident = self._resident(pipeline)

        self._add(pipeline, pending_seed("h1", stored=resident))

        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert _stored(pipeline, "h1").added_at == self._STORED_AT

    def test_resident_reacquire_refreshes_the_guard_row(self) -> None:
        # The entry's guard evidence follows the newest plan, never a frozen copy:
        # a claim joining a carried-over record re-puts the row the trust read hydrates.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        facts = GuardFacts(entry_groups=("NAN0",))

        self._add(pipeline, pending_seed("h1", guards=facts, stored=resident))

        assert _guards(pipeline) == {PENDING_AL_ID: facts}

    def test_accretion_writes_only_the_joining_claims_guard_row(self) -> None:
        # The stored claim's row is the evidence its own grab wrote. A second entry joining the record
        # never re-puts it from the hydrated copy, so a row written since the copy was read stands.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        newer = GuardFacts(entry_groups=("NewPick",))
        joining = GuardFacts(entry_groups=("NAN0",))
        pipeline.cache_store.put_guards(Arr.SONARR, PENDING_AL_ID, newer)

        self._add(pipeline, pending_seed("h1", al_id=22, guards=joining, stored=resident))

        assert _guards(pipeline) == {PENDING_AL_ID: newer, 22: joining}

    def test_accretion_replaces_the_entrys_own_claim_and_merges_the_map(self) -> None:
        # A re-flag by the entry already claiming the record: its claim is replaced
        # under its first clock, the fresh placements fold into the map, the birth stands.
        pipeline = _pipeline(torrents=self._reacquire(None))
        resident = self._resident(pipeline)
        seed = pending_seed(
            "h1", ordered_episode_ids=(101, 102), placements={"Show - 02 [1080p].mkv": [102]}, stored=resident
        )

        self._add(pipeline, seed)

        record = _stored(pipeline, "h1")
        (claim,) = record.claims
        assert claim.ordered_episode_ids == (101, 102)
        assert claim.claimed_at == self._STORED_AT
        assert record.added_at == self._STORED_AT
        assert dict(record.file_episode_map) == {
            normalize_basename("Show - 01 [1080p].mkv"): (101,),
            normalize_basename("Show - 02 [1080p].mkv"): (102,),
        }

    def test_no_add_time_born_seed_is_stamped_now(self) -> None:
        # No stored record and no usable qBittorrent time: tracked from now, birth
        # and claim alike, with the entry's guard row.
        pipeline = _pipeline(torrents=self._reacquire(None))
        before = now_stamp()

        self._add(pipeline, pending_seed("h1"))

        record = _stored(pipeline, "h1")
        assert before <= record.added_at <= now_stamp()
        assert record.claims[0].claimed_at == record.added_at
        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert pipeline._ctx.pending_imports == {}
        assert _guards(pipeline).keys() == {PENDING_AL_ID}

    def test_born_seed_joins_at_the_qbit_add_time(self) -> None:
        # qBittorrent's add time stamps the birth AND the claim, so the TTL ages the join.
        added_on = datetime.now() - timedelta(days=2)
        pipeline = _pipeline(torrents=self._reacquire(added_on))

        self._add(pipeline, pending_seed("h1"))

        record = _stored(pipeline, "h1")
        assert record.added_at == record.claims[0].claimed_at == stamp_of(added_on)
        assert pipeline._ctx.reacquired_keys == {"h1"}
        assert pipeline._ctx.pending_imports == {}

    def test_ttl_past_born_seed_is_dropped(self) -> None:
        # Past the cutoff with nothing stored: never tracked, no guard row, but
        # the outcome still surfaces so the action block reads right.
        pipeline = _pipeline(torrents=self._reacquire(datetime.now() - timedelta(days=100)))

        results = self._add(pipeline, pending_seed("h1"))

        assert _pending(pipeline) == {}
        assert pipeline._ctx.reacquired_keys == set()
        assert _guards(pipeline) == {}
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]

    def test_ttl_reads_the_configured_max_age(self) -> None:
        # The cutoff is `imports.pending_max_age_days`, not the default: five days
        # back drops under a three-day age.
        pipeline = _pipeline(
            torrents=self._reacquire(datetime.now() - timedelta(days=5)),
            import_pending_max_age_days=3,
        )

        self._add(pipeline, pending_seed("h1"))

        assert _pending(pipeline) == {}

    def test_fresh_add_never_joins_reacquired_keys(self) -> None:
        # The fresh arm is outcome-keyed: an ADDED is a this-run grab, stamped now.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "Show")}))
        before = now_stamp()

        self._add(pipeline, pending_seed("h1"))

        assert pipeline._ctx.reacquired_keys == set()
        assert [p.infohash for p in pipeline._ctx.pending_imports.values()] == ["h1"]
        assert before <= _stored(pipeline, "h1").added_at <= now_stamp()

    def test_fresh_add_of_a_stored_record_restarts_every_clock(self) -> None:
        # A re-add of a torrent whose record survived (qBittorrent lost it): the
        # accreted record enters the run list with its birth and every claim
        # stamped now, so neither the TTL nor the wait ages it from before.
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "Show")}))
        resident = self._resident(pipeline)
        before = now_stamp()

        self._add(pipeline, pending_seed("h1", al_id=22, stored=resident))

        record = _stored(pipeline, "h1")
        assert [c.al_id for c in record.claims] == [PENDING_AL_ID, 22]
        assert before <= record.added_at <= now_stamp()
        assert {c.claimed_at for c in record.claims} == {record.added_at}
        assert pipeline._ctx.reacquired_keys == set()
        assert list(pipeline._ctx.pending_imports) == ["h1"]


def _nyaa_release(*, url: str, infohash: str) -> SeadexUrlItem:
    """A download-flagged Nyaa release that clears every add-path filter."""

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return item


class TestAddTorrentCap:
    """add_torrent honors max_torrents_to_add within ONE title's url loop."""

    def test_cap_holds_the_urls_past_it(self) -> None:
        # MUTATION PIN: `cap = None` and `>= cap` -> `> cap` both over-grab. Three
        # flagged urls under a cap of 2: exactly the first two reach the service,
        # both counters read 2, and the third url is never attempted.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        u3 = _nyaa_release(url="https://nyaa.si/view/3", infohash="h3")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2, u3.url: u3})}
        torrents = FakeTorrents(
            {
                "h1": (AddOutcome.ADDED, "one"),
                "h2": (AddOutcome.ADDED, "two"),
                "h3": (AddOutcome.ADDED, "three"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=2)

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2"]  # h3 is held, never asked of qBittorrent
        assert n_added == 2
        assert pipeline._ctx.torrents_added == 2
        assert [r.outcome for r in results] == [AddOutcome.ADDED, AddOutcome.ADDED]
        assert pipeline._ctx.per_title.held_by_cap is True
        assert pipeline._ctx.stats.held_by_cap == 1

    def test_a_title_hitting_the_cap_exactly_holds_nothing(self) -> None:
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "one")}), max_torrents_to_add=1)

        n_added, _results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert n_added == 1
        assert pipeline._ctx.per_title.held_by_cap is False
        assert pipeline._ctx.stats.held_by_cap == 0

    def test_a_title_past_the_cap_is_held_and_tallied_once(self) -> None:
        # Two grabbable urls past the cap: one hold flag, one tally, no qBittorrent call.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "one"), "h2": (AddOutcome.ADDED, "two")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._ctx.torrents_added = 1

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == []
        assert (n_added, results) == (0, [])
        assert pipeline._ctx.per_title.held_by_cap is True
        assert pipeline._ctx.stats.held_by_cap == 1

    def test_a_screened_out_url_past_the_cap_is_not_held(self) -> None:
        # The private skip still lands (its flag and line), and no hold forms for a url that could not be grabbed.
        private = url_item(url="https://private.example/1", infohash="h1", download=True, is_public=False)
        seadex_dict: SeadexDict = {"RG": rg_group({private.url: private})}
        pipeline = _pipeline(torrents=FakeTorrents({}), max_torrents_to_add=1)
        pipeline._ctx.torrents_added = 1

        pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert pipeline._ctx.per_title.private_only_skipped is True
        assert pipeline._ctx.per_title.held_by_cap is False
        assert pipeline._ctx.stats.held_by_cap == 0

    def test_a_held_accreted_seed_saves_its_claim(self) -> None:
        # A torrent already downloading under a stored record keeps its mapping past the cap: the entry's claim
        # joins the record without a qBittorrent call, and nothing joins the reacquired keys.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "one")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        pipeline._ctx.torrents_added = 1
        resident = pending_import(infohash="h1", al_id=11)
        pipeline.cache_store.put_pending(Arr.SONARR, "h1", resident.to_json())
        seeds = {"h1": pending_seed("h1", al_id=22, stored=resident)}

        pipeline.add_torrent(grab_request(seadex_dict=seadex_dict, pending_seeds=seeds))

        assert torrents.calls == []
        assert claim_al_ids(_stored(pipeline, "h1")) == (11, 22)
        assert pipeline._ctx.reacquired_keys == set()
        assert pipeline._ctx.pending_imports == {}

    def test_a_held_born_seed_stores_nothing(self) -> None:
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1})}
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "one")}), max_torrents_to_add=1)
        pipeline._ctx.torrents_added = 1

        pipeline.add_torrent(grab_request(seadex_dict=seadex_dict, pending_seeds={"h1": pending_seed("h1")}))

        assert pipeline.cache_store.get_pending_record(Arr.SONARR, "h1") is None
        assert pipeline._ctx.per_title.held_by_cap is True

    def test_zero_removes_the_cap(self) -> None:
        # MUTATION PIN: without the `cap == 0` branch, 0 would flow into `>= cap`
        # and stop the run after the FIRST add instead of never.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents(
            {
                "h1": (AddOutcome.ADDED, "one"),
                "h2": (AddOutcome.ADDED, "two"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=0)

        n_added, _results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2"]
        assert n_added == 2

    def test_preview_ignores_the_cap(self) -> None:
        # A preview must walk the whole library so its report stays complete:
        # three would-be adds under a cap of 2 all go through, and no stop fires.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        u3 = _nyaa_release(url="https://nyaa.si/view/3", infohash="h3")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2, u3.url: u3})}
        torrents = FakeTorrents(
            {
                "h1": (AddOutcome.ADDED, "one"),
                "h2": (AddOutcome.ADDED, "two"),
                "h3": (AddOutcome.ADDED, "three"),
            }
        )
        pipeline = _pipeline(torrents=torrents, qbit=None, max_torrents_to_add=2)

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["h1", "h2", "h3"]
        assert n_added == 3
        assert [r.outcome for r in results] == [AddOutcome.ADDED] * 3

    def test_non_added_url_does_not_stop_the_loop(self) -> None:
        # MUTATION PIN: the non-ADDED `continue` flipped to `break` would abandon
        # the rest of the group's urls. ALREADY_ADDED first, ADDED second, ONE
        # group: both must be attempted, in order.
        u1 = _nyaa_release(url="https://nyaa.si/view/1", infohash="already")
        u2 = _nyaa_release(url="https://nyaa.si/view/2", infohash="fresh")
        seadex_dict: SeadexDict = {"RG": rg_group({u1.url: u1, u2.url: u2})}
        torrents = FakeTorrents(
            {
                "already": (AddOutcome.ALREADY_ADDED, "old"),
                "fresh": (AddOutcome.ADDED, "new"),
            }
        )
        pipeline = _pipeline(torrents=torrents)

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["already", "fresh"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED, AddOutcome.ADDED]


class TestGrabAndCacheAtTheCap:
    """The title whose add hits the cap exactly is cached done; a title with a held url is not, and still paces."""

    @staticmethod
    def _request(al_id: int, seadex_dict: SeadexDict, hashes: list[str | None]) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=seadex_dict,
            torrent_hashes=hashes,
            cache_details={"updated_at": "2026-01-01 00:00:00"},
        )

    def test_hitting_the_cap_exactly_caches_the_title(self) -> None:
        # MUTATION PIN: a "cap reached" veto in the cache gate would leave the crossing title uncached, so the
        # next run re-checked and re-grabbed it. Drive the real add path to the cap.
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1"), ["h1"]))

        assert pipeline._ctx.torrents_added == 1
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is not None
        assert pipeline._ctx.stats.needs_action == []

    def test_a_held_title_is_neither_cached_nor_a_needs_action_row(self) -> None:
        # Held is not a problem to act on: the title stays uncached so the next run grabs it, and the run keeps
        # pacing through the clock.
        clock = FakeClock()
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=3, _clock=clock)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"
        pipeline._ctx.torrents_added = 1

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1"), ["h1"]))

        assert torrents.calls == []
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert pipeline._ctx.stats.needs_action == []
        assert pipeline._ctx.stats.held_by_cap == 1
        assert clock.sleeps == [3]

    def test_a_held_title_posts_the_held_status(self) -> None:
        recording = install_recording_hub()
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")}), max_torrents_to_add=1)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.torrents_added = 1

        pipeline.grab_and_cache(self._request(42, one_release_dict(srg="RG", infohash="h1"), ["h1"]))

        statuses = [d.value.text for d in recording.of_type(EntryDetail) if d.label == "status"]
        assert statuses == ["held by the run cap; grabbed next run"]


class TestUpToDateTally:
    """The up-to-date counter accumulates across titles."""

    def test_two_up_to_date_titles_both_counted(self) -> None:
        # MUTATION PIN: `stats.up_to_date += 1` degraded to `= 1` clamps at one.
        # Two nothing-to-download titles must tally 2.
        pipeline = _pipeline(torrents=FakeTorrents({}), sleep_time=0)

        for al_id in (1, 2):
            pipeline.grab_and_cache(
                grab_request(al_id=al_id, entry=make_entry_record(url=f"https://seadex.example/{al_id}"))
            )

        assert pipeline._ctx.stats.up_to_date == 2


def _anidex_release(*, url: str, infohash: str) -> SeadexUrlItem:
    """A download-flagged release on AniDex: public (clears the private-only gate), in the default tracker set.

    It has no parser, so it hits `_add_one_url`'s new skip.
    """

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.ANIDEX
    return item


class TestUnsupportedTrackerSkip:
    """An unparseable tracker is skipped (not raised), so the id's other releases still grab.

    A title with nothing grabbable is left uncached and flagged.
    """

    def test_skipped_but_loop_continues(self) -> None:
        # AniDex first, Nyaa second, under one group. The old raise unwound the whole
        # url loop - dropping the grabbable Nyaa release too. Now AniDex is skipped and
        # the loop continues. Default config: private_releases warn, all trackers selected.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn")
        seeds = {"hn": pending_seed("hn")}

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict, pending_seeds=seeds))

        # AniDex never reached the service. Only Nyaa was handed over and added.
        assert torrents.calls == ["hn"]
        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ADDED]
        assert pipeline._ctx.per_title.unsupported_tracker_skipped is True
        assert pipeline._ctx.per_title.unsupported_tracker_groups == ["NAN0"]

    def test_unsupported_only_title_left_uncached_and_flagged(self) -> None:
        # The title's only release is on AniDex: nothing grabbable, so the title must
        # NOT be cached as done (re-checked next run) and surfaces once in needs-action.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        # Pre-seed the AniList cache so _grab's thumbnail lookup stays offline.
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["ha"],
            cache_details={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)
        assert pipeline._ctx.torrents_added == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == ["tracker not yet supported; grab manually"]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.UNSUPPORTED_TRACKER]

    def test_private_and_unsupported_surfaces_only_private(self) -> None:
        # Both a private-only skip AND an unsupported-tracker skip on one title,
        # nothing grabbed: exactly ONE needs-action reason (private-only wins) - the
        # two reasons are either/or, never both.
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        seadex_dict: SeadexDict = {"NAN0": rg_group({private.url: private, anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hp", "ha"],
            cache_details={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        # Both skips happened...
        assert pipeline._ctx.per_title.private_only_skipped is True
        assert pipeline._ctx.per_title.unsupported_tracker_skipped is True
        # ...but only the private-only reason is surfaced, and the title stays uncached.
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; private releases not supported"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY]
        assert pipeline.cache_store.get_entry(Arr.SONARR, 7) is None

    def test_private_only_in_fallback_mode_surfaces_no_alternative(self) -> None:
        # private_releases: fallback and still nothing grabbable means no public
        # alternative covered the entry's files: the needs-action row says that
        # (its own kind, so the summary tip doesn't suggest the fallback that's
        # already on).
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hp"],
            cache_details={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.per_title.private_only_skipped is True
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; no public alternative covers these files"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
        assert pipeline.cache_store.get_entry(Arr.SONARR, 7) is None

    def test_stale_held_in_fallback_mode_surfaces_stale_kind(self) -> None:
        # The planner held an owned-at-stale-size pick a fallback must not
        # replace (the stale ctx bit rides in): the needs-action row gets its
        # own kind + reason, and the title stays uncached.
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"
        pipeline._ctx.per_title.private_only_stale_held = True

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hp"],
            cache_details={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.per_title.private_only_skipped is True
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            (
                "private-only release; your copy is outdated (its file size no longer matches) "
                "and only a fallback covers it"
            )
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_STALE]
        assert pipeline.cache_store.get_entry(Arr.SONARR, 7) is None

    def test_interactive_private_pick_reads_as_a_hand_picked_no_fallback(self) -> None:
        # Interactive + fallback: a hold here is a hand-picked private pick, so
        # the reason says so - but the kind stays NO_FALLBACK so the summary tip
        # never suggests enabling the fallback that's already on.
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hp"],
            cache_details={},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "hand-picked private release; private releases not supported"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]

    def test_fallback_grab_caches_title_as_done(self) -> None:
        # The fallback happy path: the planner already unflagged the private pick
        # (public fallback kept), the fallback adds fine -> the title caches as
        # done with no needs-action row, unlike warn mode's uncached hold.
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=False)
        private.tracker = Tracker.ANIMEBYTES
        fall = url_item(url="https://nyaa.si/view/9", infohash="hf", download=True, is_fallback=True)
        fall.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {
            "Priv": rg_group({private.url: private}),
            "Fall": rg_group({fall.url: fall}),
        }

        torrents = FakeTorrents({"hf": (AddOutcome.ADDED, "Show-Fall")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hf"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)
        assert pipeline._ctx.torrents_added == 1
        assert pipeline._ctx.per_title.private_only_skipped is False
        cached = pipeline.cache_store.get_entry(Arr.SONARR, 42)
        assert cached is not None
        # A fallback grab marks the entry, so a switch to warn mode re-checks it.
        assert cached.fallback_satisfied is True
        assert pipeline.cache_store.torrent_hashes(Arr.SONARR, 42) == ["hf"]
        assert pipeline._ctx.stats.needs_action == []

    def test_mixed_grab_caches_without_the_unsupported_hash(self) -> None:
        # One grabbed (Nyaa) + one unsupported (AniDex): the title IS cached (the
        # grab completed it), but the AniDex hash is excluded from the cached set so
        # the release is re-considered on the entry's next update once a parser
        # lands. No needs-action row (something was grabbed).
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="ha")
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hn", "ha"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)
        assert pipeline._ctx.torrents_added == 1
        cached = pipeline.cache_store.get_entry(Arr.SONARR, 42)
        assert cached is not None
        # A plain (non-fallback) grab never marks the entry.
        assert cached.fallback_satisfied is False
        assert pipeline.cache_store.torrent_hashes(Arr.SONARR, 42) == ["hn"]
        assert pipeline._ctx.stats.needs_action == []

    def test_warn_mode_grab_clears_a_preseeded_marker(self) -> None:
        # A prior fallback run left fallback_satisfied=True. A later genuine grab
        # recomputes False and clears it (the marker is always written - the
        # partial-merge upsert would otherwise preserve the stale True forever).
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"Pub": rg_group({nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline.cache_store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": True})
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hn"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        cached = pipeline.cache_store.get_entry(Arr.SONARR, 7)
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_mixed_grab_keeps_the_private_hash_cached(self) -> None:
        # The private-only sibling deliberately does NOT get the exclusion:
        # private releases are never grabbed, so the private release stays
        # quietly suppressed by its cached hash.
        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({private.url: private, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            arr_title="Show",
            entry_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hn", "hp"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            replaced_groups=(),
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.per_title.private_only_skipped is True
        assert set(pipeline.cache_store.torrent_hashes(Arr.SONARR, 7)) == {"hn", "hp"}


class TestPlacementInputMissing:
    """A failed Sonarr read under a grab-time placement leaves the title uncached, with a retry row in the summary."""

    @staticmethod
    def _request(seadex_dict: SeadexDict, hashes: list[str | None]) -> GrabRequest:
        return grab_request(
            al_id=42,
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=hashes,
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            input_missing_groups=("RG",),
        )

    def test_a_grabbed_title_stays_uncached_with_a_retry_row(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        pipeline = _pipeline(torrents=FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")}), sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(seadex_dict, ["h1"]))
        assert pipeline._ctx.torrents_added == 1
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.PLACEMENT_INPUT_MISSING]
        assert rows[0].reason == "a Sonarr read the placement needs failed; will retry next run"
        assert rows[0].group == "RG"

    def test_a_title_nothing_was_flagged_for_is_neither_up_to_date_nor_cached(self) -> None:
        # The placement may have held a run, so the coverage judgment was coarse: no "already have it".
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": url_item(url="u1", infohash="h1", download=False)})}
        pipeline = _pipeline(torrents=FakeTorrents({}), sleep_time=0)
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(seadex_dict, []))

        assert pipeline._ctx.stats.up_to_date == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PLACEMENT_INPUT_MISSING]

    def test_a_failed_grab_outranks_the_missing_read(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(seadex_dict, ["h1"]))

        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.GRAB_FAILED]


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent down/erroring) is contained at the add.

    ONE clean warning (no traceback), the url loop moves on, the title stays
    uncached (retried next run), and the summary carries a GRAB_FAILED
    needs-action row instead of the title silently vanishing.
    """

    def _request(self, al_id: int, seadex_dict: SeadexDict, hashes: list[str | None]) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=seadex_dict,
            torrent_hashes=hashes,
            cache_details={"updated_at": "2026-01-01 00:00:00"},
        )

    @pytest.mark.parametrize(
        "error",
        [
            TorrentParseError("Could not find the torrent title on https://nyaa.si/view/1"),
            TorrentAddError("qBittorrent rejected the torrent"),
            httpx.ConnectError("tracker down"),
            httpx.ConnectError("nyaa down"),
            qbittorrentapi.APIConnectionError("qbit died mid-run"),
        ],
        ids=["parse", "add", "tracker", "pynyaa", "qbit"],
    )
    def test_failure_is_one_clean_warning_no_traceback(self, error: Exception) -> None:
        # Every boundary failure mode lands as ONE typed GrabFailed event (the
        # old path fell through to run_loop's per-id traceback arm. The frozen
        # fact carries no traceback by construction, and it tallies WARNING).
        torrents = FakeTorrents({}, raises={"h1": error})
        pipeline = _pipeline(torrents=torrents)
        recording = RecordingHub()
        install_hub(recording.hub)

        n_added, results = pipeline.add_torrent(
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
        # Containment is per release: the sibling url after the failure still grabs.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hbad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents)

        n_added, results = pipeline.add_torrent(grab_request(seadex_dict=seadex_dict))

        assert torrents.calls == ["hbad", "hgood"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ADDED]
        assert pipeline._ctx.per_title.grab_failed_groups == ["RG"]

    def test_failed_only_title_stays_uncached_with_a_retry_row(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict, ["h1"]))
        assert pipeline._ctx.torrents_added == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.GRAB_FAILED]
        assert rows[0].reason == "grab failed; will retry next run"
        assert rows[0].group == "RG"

    def test_partial_grab_with_a_failure_stays_uncached(self) -> None:
        # Like fallback_hold: a failure blocks the cache even when a sibling
        # grabbed, so the failed release retries next run (the add dedups).
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hbad": qbittorrentapi.APIConnectionError("qbit died")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict, ["hbad", "hgood"]))
        assert pipeline._ctx.torrents_added == 1
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.GRAB_FAILED]

    def test_cap_reached_with_a_failure_still_lands_the_grab_failed_row(self) -> None:
        # A title whose grab failed AND whose sibling add hit max_torrents_to_add
        # used to report nothing (the cap return skipped the needs-action tail):
        # the GRAB_FAILED row must still land, the run still stops, and the
        # cap-stopped title still isn't cached.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hbad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hgood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hgood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hbad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(42, seadex_dict, ["hbad", "hgood"]))

        assert pipeline._ctx.torrents_added == 1
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.GRAB_FAILED]
        assert rows[0].group == "RG"
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None

    def test_next_clean_title_caches_after_a_failed_one(self) -> None:
        # The per-title failure note lives on PerTitleState, which the prologue
        # replaces per title: title 1's failure must not hold title 2's cache
        # write hostage.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents(
            {"h2": (AddOutcome.ADDED, "Show-RG")},
            raises={"h1": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({1: {}, 2: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(1, {"RG": rg_group({bad.url: bad})}, ["h1"]))
        # The per-id prologue's fresh state between titles.
        pipeline._ctx.per_title = PerTitleState(current_title="Show S2")
        pipeline.grab_and_cache(self._request(2, {"RG": rg_group({good.url: good})}, ["h2"]))

        assert pipeline.cache_store.get_entry(Arr.SONARR, 1) is None
        assert pipeline.cache_store.get_entry(Arr.SONARR, 2) is not None


class TestFallbackHoldNeverCaches:
    """Fallback + non-interactive + a private hold: the title never caches, even on a partial grab.

    The no-fallback row resurfaces in every run's summary.
    """

    def _mixed_seadex_dict(self) -> SeadexDict:
        """A refused private group next to a grabbable public group."""

        private = url_item(url="https://ab.example/1", infohash="hp", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hn", download=True)
        nyaa.tracker = Tracker.NYAA
        return {"Priv": rg_group({private.url: private}), "Pub": rg_group({nyaa.url: nyaa})}

    def _request(self, al_id: int) -> GrabRequest:
        return grab_request(
            al_id=al_id,
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=self._mixed_seadex_dict(),
            torrent_hashes=["hn"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
        )

    def test_partial_grab_under_a_hold_stays_uncached_and_surfaces(self) -> None:
        # The public url adds fine while the private one is refused: the fallback
        # couldn't cover the private files, so despite the grab the title must NOT
        # cache (re-checked next run) and the no-fallback row must land.
        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(42))
        assert pipeline._ctx.torrents_added == 1
        assert pipeline._ctx.per_title.private_only_skipped is True
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
        # Exact wording pinned once, in test_private_only_in_fallback_mode_surfaces_no_alternative.
        assert "no public alternative" in rows[0].reason

    def test_interactive_partial_grab_still_caches(self) -> None:
        # Interactive: the hold is a hand-picked private release, so
        # the plain gate stands - the partial grab caches the title as today.
        torrents = FakeTorrents({"hn": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({43: {}})
        pipeline._ctx.per_title.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(43))

        assert pipeline._ctx.per_title.private_only_skipped is True
        assert pipeline.cache_store.get_entry(Arr.SONARR, 43) is not None
        assert pipeline._ctx.stats.needs_action == []


class TestShouldCacheAsDone:
    """The extracted cache-as-done predicate, pinned as a truth table.

    White-box: the ctx skip flags are set directly and the predicate is called
    with explicit gate inputs, so every veto axis is pinned in isolation (the
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
    ) -> bool:
        """One truth-table row. Every keyword is one axis of the predicate."""

        pipeline = make_grab_pipeline(private_releases=private_releases, interactive=interactive)
        pipeline._ctx.per_title.private_only_skipped = private_only_skipped
        pipeline._ctx.per_title.unsupported_tracker_skipped = unsupported_tracker_skipped
        pipeline._ctx.per_title.held_by_cap = held_by_cap
        if grab_failed:
            pipeline._ctx.per_title.grab_failed_groups.append("RG")
        return pipeline._should_cache_as_done(added_this_title=added_this_title)

    def test_plain_grab_caches(self) -> None:
        assert self._predicate(added_this_title=1) is True

    def test_zero_added_with_nothing_skipped_caches(self) -> None:
        # The up-to-date case: nothing grabbed because nothing was needed.
        assert self._predicate() is True

    def test_a_hold_vetoes_an_otherwise_cacheable_grab(self) -> None:
        # A url held past the cap is grabbed next run, so the title must re-check.
        assert self._predicate(added_this_title=1, held_by_cap=True) is False

    def test_fallback_hold_vetoes_despite_a_partial_grab(self) -> None:
        # The documented-surprising row: fallback mode + non-interactive + a
        # private skip vetoes caching even though something WAS grabbed - the
        # fallback couldn't cover the private files, so every run re-checks.
        assert self._predicate(private_releases="fallback", private_only_skipped=True, added_this_title=1) is False

    def test_grab_failure_vetoes_despite_a_partial_grab(self) -> None:
        assert self._predicate(added_this_title=1, grab_failed=True) is False

    def test_warn_mode_private_skip_forms_no_hold(self) -> None:
        # A mixed grab in warn mode caches (private hashes stay quietly excluded).
        assert self._predicate(private_only_skipped=True, added_this_title=1) is True

    def test_interactive_defuses_the_fallback_hold(self) -> None:
        # The hold is a hand-picked private release: plain gate stands.
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
        # Warn mode so no hold forms: the skip clause alone keeps it uncached.
        assert self._predicate(private_only_skipped=True) is False

    def test_zero_added_with_an_unsupported_tracker_skip_stays_uncached(self) -> None:
        assert self._predicate(unsupported_tracker_skipped=True) is False

    def test_mixed_unsupported_tracker_grab_caches(self) -> None:
        # The skipped hashes are excluded from the cache write, not the caching.
        assert self._predicate(unsupported_tracker_skipped=True, added_this_title=1) is True
