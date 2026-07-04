# pyright: strict
# pyright: reportPrivateUsage=false
# The add-path assertions read the pipeline's private wiring (_grab / _ctx), which
# strict re-flags; the repo disables reportPrivateUsage for tests.
"""Unit tests for the grab "produce" side (:class:`GrabPipeline`).

Pin the add path - ``_add_one_url`` registering durable :class:`PendingImport`
records, ``add_torrent``'s cap bookkeeping, and ``_grab`` returning a pure
cap-reached bool (it never finalizes; the engine owns the single finalize site).
Built bare (``object.__new__`` via ``make_bare_instance``) so no live qBittorrent
login happens; the client ``add`` is faked by ``FakeTorrents``.
"""

from collections.abc import Mapping

from seadex import Tracker

from seadexarr.modules.config import Arr
from seadexarr.modules.grab_pipeline import GrabPipeline, GrabRequest
from seadexarr.modules.manual_import import ImportWaitMode, PendingImport
from seadexarr.modules.reporter import NeedsActionKind, RunContext
from seadexarr.modules.seadex_types import SeadexDict, SeadexUrlItem
from seadexarr.modules.torrents import ReleaseOutcome

from .builders import (
    CLIENT_SENTINEL,
    AddOutcome,
    FakeTorrents,
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
    """Replaces ``GrabPipeline.add_torrent`` for the cap-return test.

    Returns a fixed ``(n_added, results)`` so ``_grab``'s cap-reached return is
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
    """A bare ``GrabPipeline`` wired for the add path (a non-preview blocking run)."""

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
    now a structural property - the pipeline can't reach ``_finalize_run`` at all;
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

        req = GrabRequest(
            al_id=1,
            item_title="Show",
            anilist_title="Show",
            sd_url="https://seadex.example/1",
            seadex_dict={},
            torrent_hashes=[],
            cache_details={},
            release_group=None,
        )

        assert pipeline._grab(req) is True


class TestAddOneUrlRegistersPending:
    """_add_one_url registers a PendingImport for a fresh AND an already-present
    torrent, but records a grab / counts toward the cap only for a fresh add."""

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
    """add_torrent honours max_torrents_to_add within ONE title's url loop."""

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
            sd_url="https://seadex.example/42",
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
                sd_url=f"https://seadex.example/{al_id}",
                seadex_dict={},
                torrent_hashes=[],
                cache_details={},
                release_group=None,
            )
            assert pipeline.grab_and_cache(req) is False

        assert pipeline._ctx.stats.up_to_date == 2


def _anidex_release(*, url: str, infohash: str) -> SeadexUrlItem:
    """A download-flagged release on AniDex - public (clears public_only) and in the
    default tracker set, but with no parser, so it hits ``_add_one_url``'s new skip."""

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.ANIDEX
    return item


class TestUnsupportedTrackerSkip:
    """An unparseable tracker is skipped (not raised), so the id's other releases
    still grab, and a title with nothing grabbable is left uncached + flagged."""

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
            sd_url="https://seadex.example/42",
            seadex_dict=seadex_dict,
            torrent_hashes=["hA"],
            cache_details={},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 0
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is None
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == ["unsupported tracker; no parser yet"]
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
            sd_url="https://seadex.example/7",
            seadex_dict=seadex_dict,
            torrent_hashes=["hP", "hA"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        # Both skips happened...
        assert pipeline._ctx.public_only_skipped is True
        assert pipeline._ctx.unsupported_tracker_skipped is True
        # ...but only the private-only reason is surfaced, and the title stays uncached.
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; private releases not allowed"
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
            sd_url="https://seadex.example/7",
            seadex_dict=seadex_dict,
            torrent_hashes=["hP"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.public_only_skipped is True
        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "private-only release; no public alternative covers these files"
        ]
        assert [r.kind for r in pipeline._ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
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
            sd_url="https://seadex.example/7",
            seadex_dict=seadex_dict,
            torrent_hashes=["hP"],
            cache_details={},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert [r.reason for r in pipeline._ctx.stats.needs_action] == [
            "hand-picked private release; private releases not allowed"
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
            sd_url="https://seadex.example/42",
            seadex_dict=seadex_dict,
            torrent_hashes=["hF"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        assert pipeline._ctx.public_only_skipped is False
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is not None
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
            sd_url="https://seadex.example/42",
            seadex_dict=seadex_dict,
            torrent_hashes=["hN", "hA"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        stop = pipeline.grab_and_cache(req)

        assert stop is False
        assert pipeline._ctx.torrents_added == 1
        assert pipeline.cache_store.get_entry(Arr.SONARR, 42) is not None
        assert pipeline.cache_store.torrent_hashes(Arr.SONARR, 42) == ["hN"]
        assert pipeline._ctx.stats.needs_action == []

    def test_mixed_grab_keeps_the_private_hash_cached(self) -> None:
        # The private-only sibling deliberately does NOT get the exclusion:
        # public_only is a user-configured exclusion, so the private release stays
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
            sd_url="https://seadex.example/7",
            seadex_dict=seadex_dict,
            torrent_hashes=["hN", "hP"],
            cache_details={"updated_at": "2026-01-01 00:00:00"},
            release_group=None,
        )

        pipeline.grab_and_cache(req)

        assert pipeline._ctx.public_only_skipped is True
        assert set(pipeline.cache_store.torrent_hashes(Arr.SONARR, 7)) == {"hN", "hP"}


class TestFallbackHoldNeverCaches:
    """fallback + non-interactive + a private hold: the title never caches - even
    on a partial grab - and the no-fallback row resurfaces in every run's summary."""

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
            sd_url=f"https://seadex.example/{al_id}",
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
        assert pipeline._ctx.public_only_skipped is True
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

        assert pipeline._ctx.public_only_skipped is True
        assert pipeline.cache_store.get_entry(Arr.SONARR, 43) is not None
        assert pipeline._ctx.stats.needs_action == []
