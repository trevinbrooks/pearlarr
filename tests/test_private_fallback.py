# pyright: strict
# pyright: reportPrivateUsage=false
# The grab assertions seed the pipeline's private AniList gateway (_anilist);
# strict re-flags that and the repo disables reportPrivateUsage for tests.
"""End-to-end regressions for ``seadex.private_releases: fallback``.

Drive the full build -> filter_downloads -> grab_and_cache path over one SeaDex
entry, pinning the fallback contract: when a preferred release is private-only,
grab the entry's best public alternative; warn only when no public alternative
exists; soft-skip (INFO, cached as done) only when the Arr genuinely already
owns the files. Regression coverage for the upgrade-pending false soft-skip
(the size-mismatch promote) and the coverage-blind per-group drop.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from seadex import EntryRecord, Tracker

from seadexarr.modules.config import Arr
from seadexarr.modules.grab_pipeline import GrabRequest
from seadexarr.modules.reporter import NeedsActionKind, RunContext
from seadexarr.modules.seadex_types import EpisodeRecord, SeadexDict

from .builders import (
    AddOutcome,
    FakeCacheStore,
    FakeTorrents,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release_filter,
    make_torrent_record,
    sonarr_ep,
)
from .fakes import CaptureHandler

PRIV_URL = "https://animebytes.tv/torrents.php?id=1"
PUB_URL = "https://nyaa.si/view/1"
PUB_HASH = "f" * 40


def _entry_private_pick_plus_public_alt() -> EntryRecord:
    """A SeaDex entry preferring a private-only release, with a public alternative."""

    priv = make_torrent_record(
        release_group="Priv",
        tracker=Tracker.ANIMEBYTES,
        url=PRIV_URL,
        infohash=None,  # private torrents carry no infohash
        file_names=("Show - S01E01.mkv",),
        file_size=999,
        is_best=True,
    )
    fall = make_torrent_record(
        release_group="Fall",
        tracker=Tracker.NYAA,
        url=PUB_URL,
        infohash=PUB_HASH,
        file_names=("Show.S01E01.web.mkv",),
        file_size=555,
        is_best=False,
    )
    return make_entry_record(anilist_id=11, torrents=(priv, fall))


def _fill_episodes(sd: SeadexDict, mapping: dict[str, list[EpisodeRecord]]) -> None:
    """Mirror ``parse_episodes_from_seadex``: per-url episodes + the group union."""

    for rg_item in sd.values():
        all_eps: list[EpisodeRecord] = []
        rg_item.all_episodes = all_eps
        for url, item in rg_item.urls.items():
            eps = list(mapping.get(url, []))
            item.episodes = eps
            all_eps.extend(eps)


@contextmanager
def _capture(logger: logging.Logger) -> Generator[CaptureHandler]:
    """Collect INFO+ records off the shared test logger, restoring it after."""

    handler = CaptureHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.WARNING)


def _grab_request(al_id: int, seadex_dict: SeadexDict, hashes: list[str | None], entry: EntryRecord) -> GrabRequest:
    """The per-id grab payload as the strategies assemble it."""

    return GrabRequest(
        al_id=al_id,
        item_title="Show",
        anilist_title="Show",
        sd_url=f"https://releases.moe/{al_id}",
        seadex_dict=seadex_dict,
        torrent_hashes=hashes,
        cache_details={"name": "Show", "updated_at": entry.updated_at},
        release_group=None,
    )


class TestUpgradePendingPromotesFallback:
    """An upgrade-pending private pick promotes the fallback, never soft-skips."""

    def test_sonarr_upgrade_pending_promotes_the_fallback(self) -> None:
        # The Arr holds Priv's release at a STALE size (the SeaDex entry updated):
        # the size-mismatch re-flag means the unflagged fallback is NOT owned, so
        # it's promoted and grabbed with the INFO keeper notice.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(public_only=True),
            cache_store=cache,
            ctx=ctx,
        )
        entry = _entry_private_pick_plus_public_alt()
        sd = filt.build(entry)
        _fill_episodes(
            sd,
            {
                PRIV_URL: [EpisodeRecord(season=1, episode=1, size=999)],
                PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
            },
        )
        ep_list = [sonarr_ep(1, 1, size=100, release_group="Priv")]

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(11, sd, {"Priv": [100]}, ep_list)

        assert out["Fall"].urls[PUB_URL].download is True
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == [PUB_HASH]
        info = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
        assert any("falling back to Fall" in m for m in info), info
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.public_only_skipped is False

        # The title caches as done only via the grab itself.
        assert cache.get_entry(Arr.SONARR, 11) is None
        torrents = FakeTorrents({PUB_HASH: (AddOutcome.ADDED, "Show S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({11: {}})
        stopped = pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert stopped is False
        assert torrents.calls == [PUB_HASH]
        assert len(ctx.stats.added) == 1
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.SONARR, 11, entry) is True
        assert cache.torrent_hashes(Arr.SONARR, 11) == [PUB_HASH]

    def test_radarr_size_disjoint_promotes_the_fallback(self) -> None:
        # The no-episode (Radarr) twin: the size-disjoint branch re-flags the
        # private pick, so the same-set fallback is promoted, not soft-skipped.
        ctx = RunContext(arr=Arr.RADARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(public_only=True),
            ctx=ctx,
        )
        sd = filt.build(_entry_private_pick_plus_public_alt())

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(22, sd, {"Priv": [100]}, None)

        assert out["Fall"].urls[PUB_URL].download is True
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == [PUB_HASH]
        info = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
        assert any("falling back to Fall" in m for m in info), info
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.public_only_skipped is False

    def test_warn_mode_upgrade_pending_warns_and_holds(self) -> None:
        # Parity: warn mode on the same upgrade-pending state warns and leaves
        # the title uncached, surfacing the plain PRIVATE_ONLY needs-action row.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(public_only=True),
            cache_store=cache,
            ctx=ctx,
        )
        entry = _entry_private_pick_plus_public_alt()
        sd = filt.build(entry)
        assert set(sd) == {"Priv"}  # warn mode adds no fallback

        _fill_episodes(sd, {PRIV_URL: [EpisodeRecord(season=1, episode=1, size=999)]})
        ep_list = [sonarr_ep(1, 1, size=100, release_group="Priv")]

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(11, sd, {"Priv": [100]}, ep_list)

        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any("private-only (private releases not allowed)" in m for m in warnings), warnings
        assert ctx.public_only_skipped is True

        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="warn", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert cache.get_entry(Arr.SONARR, 11) is None
        assert [n.kind for n in ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY]


class TestOwnedFallbackSoftSkip:
    """The legitimate soft-skip survives: the Arr genuinely owns the files."""

    def test_hash_mode_cached_fallback_still_soft_skips(self) -> None:
        # Hash mode: the fallback's hash is cached (a prior run grabbed it) while
        # the hashless private pick re-flags off the cache's None gap. Hash mode
        # never sets the size-mismatch marker, so the soft-skip fires at INFO and
        # the title still caches as done.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 11, {"torrent_hashes": [PUB_HASH]})
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(public_only=True, use_torrent_hash_to_filter=True),
            cache_store=cache,
            ctx=ctx,
        )
        entry = _entry_private_pick_plus_public_alt()
        sd = filt.build(entry)

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(11, sd, {}, None)

        assert out["Priv"].urls[PRIV_URL].download is False
        assert out["Fall"].urls[PUB_URL].download is False
        assert set(hashes) == {None, PUB_HASH}
        info = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
        assert any("a public fallback already covers these files" in m for m in info), info
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.public_only_skipped is False

        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="fallback", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert ctx.stats.up_to_date == 1
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.SONARR, 11, entry) is True


class TestSurvivingPrivateCoverage:
    """A private url kept for uncovered files is warned about, never silently lost."""

    def test_surviving_private_batch_warns_at_add_time(self) -> None:
        # One group: a private S1+S2 batch survives the coverage-aware drop next
        # to its public S1-only fallback. With both episodes missing, the batch
        # stays flagged and the add-time gate warns + sets the hold flag - the
        # old drop lost the S2 coverage without a trace.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=True,
            planner=make_planner(public_only=True),
            cache_store=cache,
            ctx=ctx,
        )
        priv_batch = make_torrent_record(
            release_group="A",
            tracker=Tracker.ANIMEBYTES,
            url=PRIV_URL,
            infohash=None,
            file_names=("A - S01E01.mkv", "A - S02E01.mkv"),
            file_size=999,
            is_dual_audio=True,
            is_best=True,
        )
        pub_s1 = make_torrent_record(
            release_group="A",
            tracker=Tracker.NYAA,
            url=PUB_URL,
            infohash=PUB_HASH,
            file_names=("A.S01E01.web.mkv",),
            file_size=555,
            is_best=False,
        )
        entry = make_entry_record(anilist_id=33, torrents=(priv_batch, pub_s1))
        sd = filt.build(entry)
        assert set(sd["A"].urls) == {PRIV_URL, PUB_URL}

        _fill_episodes(
            sd,
            {
                PRIV_URL: [
                    EpisodeRecord(season=1, episode=1, size=999),
                    EpisodeRecord(season=2, episode=1, size=999),
                ],
                PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
            },
        )

        # Sonarr is missing BOTH episodes: the batch and the fallback both flag.
        hashes, out = filt.filter_downloads(33, sd, {}, [sonarr_ep(1, 1), sonarr_ep(2, 1)])
        assert out["A"].urls[PRIV_URL].download is True
        assert out["A"].urls[PUB_URL].download is True
        assert hashes == [PUB_HASH]

        torrents = FakeTorrents({PUB_HASH: (AddOutcome.ADDED, "A S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({33: {}})
        with _capture(pipe.log_fmt.logger) as handler:
            pipe.grab_and_cache(_grab_request(33, out, hashes, entry))

        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any("private-only (private releases not allowed)" in m for m in warnings), warnings
        assert ctx.public_only_skipped is True
        assert ctx.public_only_groups == ["A"]
        assert torrents.calls == [PUB_HASH]
        # The mixed-title gate still caches (something was grabbed); the add-time
        # warning is the surfaced signal, so no needs-action row.
        assert cache.check_al_id_in_cache(Arr.SONARR, 33, entry) is True
        assert ctx.stats.needs_action == []
