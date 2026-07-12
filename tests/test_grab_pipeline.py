# pyright: strict
# pyright: reportPrivateUsage=false
# The add-path assertions read the pipeline's private wiring (_grab / _ctx), which
# strict re-flags; the repo disables reportPrivateUsage for tests.
"""Unit tests for the grab "produce" side (`GrabPipeline`).

Pin the add path - `_add_one_url` registering durable `PendingImport`
records, `add_torrent`'s cap bookkeeping, and `_grab` returning a pure
cap-reached bool (it never finalizes; the engine owns the single finalize site).
Built bare (`object.__new__` via `make_bare_instance`) so no live qBittorrent
login happens; the client `add` is faked by `FakeTorrents`.
"""

from collections.abc import Mapping

import httpx
import pytest
import qbittorrentapi
from seadex import Tracker

from pearlarr import notify
from pearlarr.config import Arr
from pearlarr.discord import DiscordEmbed
from pearlarr.grab_pipeline import GrabPipeline, GrabRequest
from pearlarr.manual_import import ImportWaitMode, PendingImport
from pearlarr.notify import Notifier
from pearlarr.output import GrabFailed, Severity, install_hub, severity_of
from pearlarr.output.recording import RecordingHub
from pearlarr.reporter import NeedsActionKind, RunContext
from pearlarr.seadex_types import SeadexDict, SeadexUrlItem
from pearlarr.torrent import TorrentParseError
from pearlarr.torrents import ReleaseOutcome, TorrentAddError

from .builders import (
    CLIENT_SENTINEL,
    AddOutcome,
    FakeTorrents,
    make_entry_record,
    make_grab_pipeline,
    one_release_dict,
    pending_import,
    rg_group,
    url_item,
)


def _stub_add_torrent(
    torrent_dict: SeadexDict,
    pending_seeds: dict[str, PendingImport] | None = None,
) -> tuple[int, list[ReleaseOutcome]]:
    """Replaces `GrabPipeline.add_torrent` for the cap-return test.

    Returns a fixed `(n_added, results)` so `_grab`'s cap-reached return is
    exercised without a real qBittorrent add - the bool the engine's single
    finalize site keys off.
    """

    del torrent_dict, pending_seeds
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


def _pending(pipeline: GrabPipeline) -> Mapping[str, object]:
    """The pipeline's durable per-arr pending store (what the engine reads back)."""

    return pipeline.cache_store.get_pending(Arr.SONARR)


class TestGrabReturnsPureBool:
    """_grab signals cap-reached as a bool and never finalizes itself.

    GrabPipeline holds no reference back to the engine, so "without finalizing" is
    now a structural property - the pipeline can't reach `_finalize_run` at all;
    the test pins the cap-reached return value the engine's single finalize site
    keys off.
    """

    def test_grab_at_cap_returns_true(self) -> None:
        pipeline = make_grab_pipeline(
            qbit=None,
            max_torrents_to_add=1,
            add_torrent=_stub_add_torrent,
        )
        pipeline._ctx.torrents_added = 1  # already at the cap of 1
        # Warm the gateway cache so the embed's thumb lookup never hits AniList.
        pipeline._anilist.al_cache.update({1: {}})

        req = GrabRequest(
            al_id=1,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/1"),
            seadex_dict={},
            torrent_hashes=[],
            cache_details={},
            release_group=None,
        )

        assert pipeline._grab(req) is True


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
                item_title="Show",
                anilist_title="Show Title",
                entry=make_entry_record(url="https://releases.moe/7", notes="the why"),
                seadex_dict=one_release_dict(srg="PMR", infohash="h1"),
                torrent_hashes=["h1"],
                cache_details={},
                release_group=["OldGroup"],
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
        # A single-group grab hoists its pick into the description; the
        # subtitle/notes stack trails as the nameless (header-free) field.
        assert embed.description == "**Grabbed · `PMR`**\n[Nyaa](https://nyaa.si/view/1)"
        assert [f.name for f in embed.fields] == ["Episodes", "Replacing", ""]
        assert embed.fields[-1].value == "-# Show\n> the why"

    def test_nothing_added_pushes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._grab(monkeypatch, outcome=AddOutcome.ALREADY_ADDED) == []

    def test_preview_never_pushes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A preview simulates the add (so the count is non-zero) but the push is
        # an outward notification and must stay silent.
        assert self._grab(monkeypatch, qbit=None) == []


class TestAddOneUrlRegistersPending:
    """`_add_one_url` registers a `PendingImport` for both a fresh and an already-present torrent.

    It records a grab and counts toward the cap only for a fresh add.
    """

    def test_already_added_registers_pending_import(self) -> None:
        # The recommended release is already in qBittorrent (a prior run grabbed
        # it, still downloading): register it for the monitor, but don't count it
        # as a this-run grab.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        seeds = {"h1": pending_import(infohash="h1", series_id=7)}

        n_added, results = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert [p.infohash for p in pipeline._ctx.pending_imports] == ["h1"]
        assert n_added == 0
        assert pipeline._ctx.torrents_added == 0
        assert pipeline._ctx.stats.added == []
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        seeds = {"h1": pending_import(infohash="h1")}

        n_added, _ = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert [p.infohash for p in pipeline._ctx.pending_imports] == ["h1"]
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
        seeds = {
            "already": pending_import(infohash="already"),
            "fresh": pending_import(infohash="fresh"),
        }

        n_added, _ = pipeline.add_torrent(seadex_dict, pending_seeds=seeds)

        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1

    def test_no_seed_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents)

        pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds={},
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []

    def test_off_mode_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, mode=ImportWaitMode.OFF)
        seeds = {"h1": pending_import(infohash="h1")}

        pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []

    def test_preview_does_not_register_but_returns_outcome(self) -> None:
        # No client -> preview: nothing persisted, but the outcome still surfaces.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, qbit=None)
        seeds = {"h1": pending_import(infohash="h1")}

        _, results = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]


def _nyaa_release(*, url: str, infohash: str) -> SeadexUrlItem:
    """A download-flagged Nyaa release that clears every add-path filter."""

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return item


class TestAddTorrentCap:
    """add_torrent honors max_torrents_to_add within ONE title's url loop."""

    def test_cap_stops_after_exactly_cap_adds(self) -> None:
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

        n_added, results = pipeline.add_torrent(seadex_dict, pending_seeds=None)

        assert torrents.calls == ["h1", "h2"]  # early stop: h3 never attempted
        assert n_added == 2
        assert pipeline._ctx.torrents_added == 2
        assert [r.outcome for r in results] == [AddOutcome.ADDED, AddOutcome.ADDED]

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

        n_added, results = pipeline.add_torrent(seadex_dict, pending_seeds=None)

        assert torrents.calls == ["already", "fresh"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED, AddOutcome.ADDED]


class TestGrabAndCacheCapStop:
    """grab_and_cache propagates the cap stop: True out, and NO cache write."""

    def test_cap_stop_returns_true_and_skips_the_cache_write(self) -> None:
        # MUTATION PIN: the cap branch's `return True` flipped to `False` would
        # fall through to the per-title cache update - caching a title mid-cap -
        # and tell the engine to keep scanning. Drive the real add path to the cap.
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-RG")})
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=one_release_dict(srg="RG", infohash="h1"),
            torrent_hashes=["h1"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is True
        assert pipeline._ctx.torrents_added == 1
        # The engine's single finalize site owns the save; no per-title write here.
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        # A clean cap stop reports nothing extra (no phantom needs-action row).
        assert pipeline._ctx.stats.needs_action == []


class TestUpToDateTally:
    """The up-to-date counter accumulates across titles."""

    def test_two_up_to_date_titles_both_counted(self) -> None:
        # MUTATION PIN: `stats.up_to_date += 1` degraded to `= 1` clamps at one;
        # two nothing-to-download titles must tally 2.
        pipeline = _pipeline(torrents=FakeTorrents({}), sleep_time=0)

        for al_id in (1, 2):
            req = GrabRequest(
                al_id=al_id,
                item_title="Show",
                anilist_title="Show",
                entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
                seadex_dict={},
                torrent_hashes=[],
                cache_details={},
                release_group=None,
            )
            assert pipeline.grab_and_cache(req) is False

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
        # url loop - dropping the grabbable Nyaa release too; now AniDex is skipped and
        # the loop continues. Default config: private_releases warn, all trackers selected.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="hA")
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hN", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn")
        seeds = {"hN": pending_import(infohash="hN", series_id=7)}

        n_added, results = pipeline.add_torrent(seadex_dict, pending_seeds=seeds)

        # AniDex never reached the service; only Nyaa was handed over and added.
        assert torrents.calls == ["hN"]
        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ADDED]
        assert pipeline._ctx.unsupported_tracker_skipped is True
        assert pipeline._ctx.unsupported_tracker_groups == ["NAN0"]

    def test_unsupported_only_title_left_uncached_and_flagged(self) -> None:
        # The title's only release is on AniDex: nothing grabbable, so the title must
        # NOT be cached as done (re-checked next run) and surfaces once in needs-action.
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="hA")
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        # Pre-seed the AniList cache so _grab's thumbnail lookup stays offline.
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hA"],
            cache_details={},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == ["tracker not yet supported; grab manually"]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.UNSUPPORTED_TRACKER]

    def test_private_and_unsupported_surfaces_only_private(self) -> None:
        # Both a private-only skip AND an unsupported-tracker skip on one title,
        # nothing grabbed: exactly ONE needs-action reason (private-only wins) - the
        # two reasons are either/or, never both.
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="hA")
        seadex_dict: SeadexDict = {"NAN0": rg_group({private.url: private, anidex.url: anidex})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hP", "hA"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        # Both skips happened...
        assert pipeline._ctx.private_only_skipped is True
        assert pipeline._ctx.unsupported_tracker_skipped is True
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
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hP"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.private_only_skipped is True
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; no public alternative covers these files"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
        assert pipeline.cache_store.get_entry(Arr.SONARR, 7) is None

    def test_stale_held_in_fallback_mode_surfaces_stale_kind(self) -> None:
        # The planner held an owned-at-stale-size pick a fallback must not
        # replace (the stale ctx bit rides in): the needs-action row gets its
        # own kind + reason, and the title stays uncached.
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"
        pipeline._ctx.private_only_stale_held = True

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hP"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.private_only_skipped is True
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; your copy is outdated (its file size no longer matches) "
            "and only a fallback covers it"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_STALE]
        assert pipeline.cache_store.get_entry(Arr.SONARR, 7) is None

    def test_interactive_private_pick_reads_as_a_hand_picked_no_fallback(self) -> None:
        # Interactive + fallback: a hold here is the user's own private pick, so
        # the reason says so - but the kind stays NO_FALLBACK so the summary tip
        # never suggests enabling the fallback that's already on.
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        seadex_dict: SeadexDict = {"Priv": rg_group({private.url: private})}

        pipeline = _pipeline(torrents=FakeTorrents({}), private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hP"],
            cache_details={},
            release_group=None,
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
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=False)
        private.tracker = Tracker.ANIMEBYTES
        fall = url_item(url="https://nyaa.si/view/9", infohash="hF", download=True, is_fallback=True)
        fall.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {
            "Priv": rg_group({private.url: private}),
            "Fall": rg_group({fall.url: fall}),
        }

        torrents = FakeTorrents({"hF": (AddOutcome.ADDED, "Show-Fall")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hF"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        assert pipeline._ctx.private_only_skipped is False
        cached = pipeline.cache_store.get_entry(Arr.SONARR, 42)
        assert cached is not None
        # A fallback grab marks the entry, so a switch to warn mode re-checks it.
        assert cached.fallback_satisfied is True
        assert pipeline.cache_store.torrent_hashes(Arr.SONARR, 42) == ["hF"]
        assert pipeline._ctx.stats.needs_action == []

    def test_mixed_grab_caches_without_the_unsupported_hash(self) -> None:
        # One grabbed (Nyaa) + one unsupported (AniDex): the title IS cached (the
        # grab completed it), but the AniDex hash is excluded from the cached set so
        # the release is re-considered on the entry's next update once a parser
        # lands. No needs-action row (something was grabbed).
        anidex = _anidex_release(url="https://anidex.info/torrent/1", infohash="hA")
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hN", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({anidex.url: anidex, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=42,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/42"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hN", "hA"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        cached = pipeline.cache_store.get_entry(Arr.SONARR, 42)
        assert cached is not None
        # A plain (non-fallback) grab never marks the entry.
        assert cached.fallback_satisfied is False
        assert pipeline.cache_store.torrent_hashes(Arr.SONARR, 42) == ["hN"]
        assert pipeline._ctx.stats.needs_action == []

    def test_warn_mode_grab_clears_a_preseeded_marker(self) -> None:
        # A prior fallback run left fallback_satisfied=True; a later genuine grab
        # recomputes False and clears it (the marker is always written - the
        # partial-merge upsert would otherwise preserve the stale True forever).
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hN", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"Pub": rg_group({nyaa.url: nyaa})}

        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline.cache_store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": True})
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hN"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        cached = pipeline.cache_store.get_entry(Arr.SONARR, 7)
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_mixed_grab_keeps_the_private_hash_cached(self) -> None:
        # The private-only sibling deliberately does NOT get the exclusion:
        # private releases are never grabbed, so the private release stays
        # quietly suppressed by its cached hash.
        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hN", download=True)
        nyaa.tracker = Tracker.NYAA
        seadex_dict: SeadexDict = {"NAN0": rg_group({private.url: private, nyaa.url: nyaa})}

        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents, private_releases="warn", sleep_time=0)
        pipeline._anilist.al_cache.update({7: {}})
        pipeline._ctx.current_title = "Show S1"

        req = GrabRequest(
            al_id=7,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url="https://seadex.example/7"),
            seadex_dict=seadex_dict,
            torrent_hashes=["hN", "hP"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.private_only_skipped is True
        assert set(pipeline.cache_store.torrent_hashes(Arr.SONARR, 7)) == {"hN", "hP"}


class TestGrabFailureContainment:
    """An expected external failure (tracker or qBittorrent down/erroring) is contained at the add.

    ONE clean warning (no traceback), the url loop moves on, the title stays
    uncached (retried next run), and the summary carries a GRAB_FAILED
    needs-action row instead of the title silently vanishing.
    """

    def _request(self, al_id: int, seadex_dict: SeadexDict, hashes: list[str | None]) -> GrabRequest:
        return GrabRequest(
            al_id=al_id,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=seadex_dict,
            torrent_hashes=hashes,
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
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
        # old path fell through to run_loop's per-id traceback arm; the frozen
        # fact carries no traceback by construction, and it tallies WARNING).
        torrents = FakeTorrents({}, raises={"h1": error})
        pipeline = _pipeline(torrents=torrents)
        recording = RecordingHub()
        install_hub(recording.hub)

        n_added, results = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=None,
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
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hBad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hGood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hGood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hBad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents)

        n_added, results = pipeline.add_torrent(seadex_dict, pending_seeds=None)

        assert torrents.calls == ["hBad", "hGood"]
        assert n_added == 1
        assert [r.outcome for r in results] == [AddOutcome.ADDED]
        assert pipeline._grab_failed_groups == ["RG"]

    def test_failed_only_title_stays_uncached_with_a_retry_row(self) -> None:
        nyaa = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        seadex_dict: SeadexDict = {"RG": rg_group({nyaa.url: nyaa})}
        torrents = FakeTorrents({}, raises={"h1": httpx.ConnectError("nyaa down")})
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        stop = pipeline.grab_and_cache(self._request(42, seadex_dict, ["h1"]))

        assert stop is False
        assert pipeline._ctx.torrents_added == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.GRAB_FAILED]
        assert rows[0].reason == "grab failed; will retry next run"
        assert rows[0].group == "RG"

    def test_partial_grab_with_a_failure_stays_uncached(self) -> None:
        # Like fallback_hold: a failure blocks the cache even when a sibling
        # grabbed, so the failed release retries next run (the add dedups).
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hBad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hGood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hGood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hBad": qbittorrentapi.APIConnectionError("qbit died")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        stop = pipeline.grab_and_cache(self._request(42, seadex_dict, ["hBad", "hGood"]))

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.GRAB_FAILED]

    def test_cap_reached_with_a_failure_still_lands_the_grab_failed_row(self) -> None:
        # A title whose grab failed AND whose sibling add hit max_torrents_to_add
        # used to report nothing (the cap return skipped the needs-action tail):
        # the GRAB_FAILED row must still land, the run still stops, and the
        # cap-stopped title still isn't cached.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="hBad")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="hGood")
        seadex_dict: SeadexDict = {"RG": rg_group({bad.url: bad, good.url: good})}
        torrents = FakeTorrents(
            {"hGood": (AddOutcome.ADDED, "Show-RG")},
            raises={"hBad": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1, sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        stop = pipeline.grab_and_cache(self._request(42, seadex_dict, ["hBad", "hGood"]))

        assert stop is True  # the cap still stops the run
        assert pipeline._ctx.torrents_added == 1
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.GRAB_FAILED]
        assert rows[0].group == "RG"
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None

    def test_next_clean_title_caches_after_a_failed_one(self) -> None:
        # The per-title failure note resets: title 1's failure must not hold
        # title 2's cache write hostage.
        bad = _nyaa_release(url="https://nyaa.si/view/1", infohash="h1")
        good = _nyaa_release(url="https://nyaa.si/view/2", infohash="h2")
        torrents = FakeTorrents(
            {"h2": (AddOutcome.ADDED, "Show-RG")},
            raises={"h1": httpx.ConnectError("nyaa down")},
        )
        pipeline = _pipeline(torrents=torrents, sleep_time=0)
        pipeline._anilist.al_cache.update({1: {}, 2: {}})
        pipeline._ctx.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(1, {"RG": rg_group({bad.url: bad})}, ["h1"]))
        pipeline.grab_and_cache(self._request(2, {"RG": rg_group({good.url: good})}, ["h2"]))

        assert pipeline.cache_store.get_entry(Arr.SONARR, 1) is None
        assert pipeline.cache_store.get_entry(Arr.SONARR, 2) is not None


class TestFallbackHoldNeverCaches:
    """Fallback + non-interactive + a private hold: the title never caches, even on a partial grab.

    The no-fallback row resurfaces in every run's summary.
    """

    def _mixed_seadex_dict(self) -> SeadexDict:
        """A refused private group next to a grabbable public group."""

        private = url_item(url="https://ab.example/1", infohash="hP", is_public=False, download=True)
        private.tracker = Tracker.ANIMEBYTES
        nyaa = url_item(url="https://nyaa.si/view/2", infohash="hN", download=True)
        nyaa.tracker = Tracker.NYAA
        return {"Priv": rg_group({private.url: private}), "Pub": rg_group({nyaa.url: nyaa})}

    def _request(self, al_id: int) -> GrabRequest:
        return GrabRequest(
            al_id=al_id,
            item_title="Show",
            anilist_title="Show",
            entry=make_entry_record(url=f"https://seadex.example/{al_id}"),
            seadex_dict=self._mixed_seadex_dict(),
            torrent_hashes=["hN"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

    def test_partial_grab_under_a_hold_stays_uncached_and_surfaces(self) -> None:
        # The public url adds fine while the private one is refused: the fallback
        # couldn't cover the private files, so despite the grab the title must NOT
        # cache (re-checked next run) and the no-fallback row must land.
        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", sleep_time=0)
        pipeline._anilist.al_cache.update({42: {}})
        pipeline._ctx.current_title = "Show S1"

        stop = pipeline.grab_and_cache(self._request(42))

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        assert pipeline._ctx.private_only_skipped is True
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        rows = pipeline._ctx.stats.needs_action
        assert [r.kind for r in rows] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
        # Exact wording pinned once, in test_private_only_in_fallback_mode_surfaces_no_alternative.
        assert "no public alternative" in rows[0].reason

    def test_interactive_partial_grab_still_caches(self) -> None:
        # Interactive: the hold is the user's own hand-picked private release, so
        # the plain gate stands - the partial grab caches the title as today.
        torrents = FakeTorrents({"hN": (AddOutcome.ADDED, "Show-Pub")})
        pipeline = _pipeline(torrents=torrents, private_releases="fallback", interactive=True, sleep_time=0)
        pipeline._anilist.al_cache.update({43: {}})
        pipeline._ctx.current_title = "Show S1"

        pipeline.grab_and_cache(self._request(43))

        assert pipeline._ctx.private_only_skipped is True
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
        cap_reached: bool = False,
        added_this_title: int = 0,
        grab_failed: bool = False,
    ) -> bool:
        """One truth-table row; every keyword is one axis of the predicate."""

        pipeline = make_grab_pipeline(private_releases=private_releases, interactive=interactive)
        pipeline._ctx.private_only_skipped = private_only_skipped
        pipeline._ctx.unsupported_tracker_skipped = unsupported_tracker_skipped
        return pipeline._should_cache_as_done(
            cap_reached=cap_reached,
            added_this_title=added_this_title,
            grab_failed=grab_failed,
        )

    def test_plain_grab_caches(self) -> None:
        assert self._predicate(added_this_title=1) is True

    def test_zero_added_with_nothing_skipped_caches(self) -> None:
        # The up-to-date case: nothing grabbed because nothing was needed.
        assert self._predicate() is True

    def test_cap_vetoes_an_otherwise_cacheable_grab(self) -> None:
        # The cap can stop the url loop mid-title, leaving later urls unattempted.
        assert self._predicate(added_this_title=1, cap_reached=True) is False

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
        # The hold is the user's own hand-picked private release: plain gate stands.
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
