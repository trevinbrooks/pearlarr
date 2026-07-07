# pyright: strict
# pyright: reportPrivateUsage=false
# The grab assertions seed the pipeline's private AniList gateway (_anilist);
# strict re-flags that and the repo disables reportPrivateUsage for tests.
"""End-to-end regressions for ``seadex.private_releases: fallback``.

Drive the full build -> filter_downloads -> grab_and_cache path over one SeaDex
entry, pinning the fallback contract: when a preferred release is private-only,
grab the entry's best public alternative; warn when no public alternative
exists, or when the Arr owns the preferred private release at a stale size (a
fallback never replaces an owned copy); soft-skip (INFO, cached as done) only
when the Arr genuinely already owns the files. Regression coverage for the
preferred-public size-mismatch promote and the coverage-blind per-group drop.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from seadex import EntryRecord, TorrentRecord, Tracker

from seadexarr.modules.config import Arr
from seadexarr.modules.grab_pipeline import GrabRequest
from seadexarr.modules.log import EntryState
from seadexarr.modules.reporter import NeedsActionKind, RunContext
from seadexarr.modules.seadex_types import EpisodeRecord, SeadexDict

from .builders import (
    AddOutcome,
    FakeCacheStore,
    FakeSeaDexSource,
    FakeTorrents,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release_filter,
    make_services,
    make_torrent_record,
    sonarr_ep,
)
from .fakes import CaptureHandler

PRIV_URL = "https://animebytes.tv/torrents.php?id=1"
PUB_URL = "https://nyaa.si/view/1"
PUB_HASH = "f" * 40

# The stale-owned hold's planner notice and needs-action row wording, pinned.
STALE_NOTICE = (
    "private-only; your copy is outdated (its file size no longer matches the release) and only a fallback covers it"
)
STALE_ROW_REASON = (
    "private-only release; your copy is outdated (its file size no longer matches) and only a fallback covers it"
)


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


class TestUpgradePendingHoldsForOwnedStale:
    """An owned-at-stale-size private pick warns and holds - a fallback never replaces it."""

    def test_sonarr_upgrade_pending_warns_and_holds(self) -> None:
        # The Arr holds Priv's release at a STALE size (the SeaDex entry updated):
        # a fallback (a cascade loser) must not supersede the owned preferred
        # copy, so nothing is promoted and the set warns with the stale reason.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
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

        assert out["Fall"].urls[PUB_URL].download is False
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == []
        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any(STALE_NOTICE in m for m in warnings), warnings
        assert ctx.private_only_skipped is True
        assert ctx.private_only_stale_held is True

        # The fallback hold keeps the title uncached and surfaces the STALE row,
        # so it resurfaces every run until the user updates or deletes the copy.
        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="fallback", sleep_time=0)
        stopped = pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert stopped is False
        assert cache.get_entry(Arr.SONARR, 11) is None
        assert [n.kind for n in ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_STALE]
        assert [n.reason for n in ctx.stats.needs_action] == [STALE_ROW_REASON]

    def test_radarr_size_disjoint_warns_and_holds(self) -> None:
        # The no-episode (Radarr) twin: the size-disjoint branch re-flags the
        # private pick, and the same-set fallback stays un-promoted - hold.
        ctx = RunContext(arr=Arr.RADARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            ctx=ctx,
        )
        sd = filt.build(_entry_private_pick_plus_public_alt())

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(22, sd, {"Priv": [100]}, None)

        assert out["Fall"].urls[PUB_URL].download is False
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == []
        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any(STALE_NOTICE in m for m in warnings), warnings
        assert ctx.private_only_skipped is True
        assert ctx.private_only_stale_held is True


class TestOwnedPreferredPrivateAtMatchingSize:
    """The Arr owns the preferred private release at SeaDex's size: fully protected.

    The matcher never flags the private pick (group + size match) NOR the riding
    fallback (another recommended group already covers its files), so the title
    reads up-to-date and caches as done - no notices, no needs-action rows.
    Also mutation-kills the two matcher already-covered gates.
    """

    def test_sonarr_owned_private_at_matching_size_stays_untouched(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
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
        # Sonarr holds Priv's release at the MATCHING SeaDex size.
        ep_list = [sonarr_ep(1, 1, size=999, release_group="Priv")]

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(11, sd, {"Priv": [999]}, ep_list)

        assert out["Priv"].urls[PRIV_URL].download is False
        assert out["Fall"].urls[PUB_URL].download is False
        assert hashes == []
        assert handler.records == []
        assert ctx.private_only_skipped is False
        assert ctx.private_only_stale_held is False

        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="fallback", sleep_time=0)
        stopped = pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert stopped is False
        assert ctx.stats.up_to_date == 1
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.SONARR, 11, entry) is True
        cached = cache.get_entry(Arr.SONARR, 11)
        assert cached is not None
        # The owned PREFERRED private pick satisfied the title, not the idle
        # fallback pool - the false positive the coarse any(is_fallback) had.
        assert cached.fallback_satisfied is False

    def test_radarr_owned_private_at_matching_size_stays_untouched(self) -> None:
        # The no-episode (Radarr) twin, via the arr_release_dict size overlap.
        ctx = RunContext(arr=Arr.RADARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            cache_store=cache,
            ctx=ctx,
        )
        entry = _entry_private_pick_plus_public_alt()
        sd = filt.build(entry)

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(22, sd, {"Priv": [999]}, None)

        assert out["Priv"].urls[PRIV_URL].download is False
        assert out["Fall"].urls[PUB_URL].download is False
        assert hashes == []
        assert handler.records == []
        assert ctx.private_only_skipped is False
        assert ctx.private_only_stale_held is False

        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="fallback", sleep_time=0)
        stopped = pipe.grab_and_cache(_grab_request(22, out, hashes, entry))

        assert stopped is False
        assert ctx.stats.up_to_date == 1
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.RADARR, 22, entry) is True
        cached = cache.get_entry(Arr.RADARR, 22)
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_warn_mode_upgrade_pending_warns_and_holds(self) -> None:
        # Parity: warn mode on the same upgrade-pending state warns and leaves
        # the title uncached, surfacing the plain PRIVATE_ONLY needs-action row.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
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
        assert any("private-only (private releases not supported)" in m for m in warnings), warnings
        assert ctx.private_only_skipped is True

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
            planner=make_planner(use_torrent_hash_to_filter=True),
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
        assert ctx.private_only_skipped is False
        assert ctx.fallback_covered is True

        pipe = make_grab_pipeline(cache_store=cache, _ctx=ctx, private_releases="fallback", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, out, hashes, entry))

        assert ctx.stats.up_to_date == 1
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.SONARR, 11, entry) is True
        cached = cache.get_entry(Arr.SONARR, 11)
        assert cached is not None
        # The owned-fallback soft-skip marks the cache, so warn mode re-checks it.
        assert cached.fallback_satisfied is True


class TestFilterDownloadsNoticeSeam:
    """filter_downloads renders each SkipNotice at its level and stamps the ctx."""

    PRIVA_URL = "https://animebytes.tv/torrents.php?id=10"
    PRIVB_URL = "https://animebytes.tv/torrents.php?id=20"
    PUBA_URL = "https://nyaa.si/view/9"
    PUBA_HASH = "e" * 40

    def test_renders_both_levels_and_carries_skip_state(self) -> None:
        # One entry, two same-files sets: {PrivA, PubA} promotes the preferred
        # public group (an INFO notice); {PrivB} is private-only with no cover
        # (a WARNING notice, plus the skip flag + group carried onto the
        # RunContext).
        ctx = RunContext(arr=Arr.SONARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            ctx=ctx,
        )
        priv_a = make_torrent_record(
            release_group="PrivA",
            tracker=Tracker.ANIMEBYTES,
            url=self.PRIVA_URL,
            infohash=None,
            file_names=("A - S01E01.mkv",),
            file_size=999,
            is_best=True,
        )
        priv_b = make_torrent_record(
            release_group="PrivB",
            tracker=Tracker.ANIMEBYTES,
            url=self.PRIVB_URL,
            infohash=None,
            file_names=("B - S02E01.mkv",),
            file_size=777,
            is_best=True,
        )
        pub_a = make_torrent_record(
            release_group="PubA",
            tracker=Tracker.NYAA,
            url=self.PUBA_URL,
            infohash=self.PUBA_HASH,
            file_names=("A.S01E01.web.mkv",),
            file_size=555,
            is_best=True,
        )
        sd = filt.build(make_entry_record(anilist_id=44, torrents=(priv_a, priv_b, pub_a)))
        assert set(sd) == {"PrivA", "PrivB", "PubA"}
        _fill_episodes(
            sd,
            {
                self.PRIVA_URL: [EpisodeRecord(season=1, episode=1, size=999)],
                self.PRIVB_URL: [EpisodeRecord(season=2, episode=1, size=777)],
                self.PUBA_URL: [EpisodeRecord(season=1, episode=1, size=555)],
            },
        )
        # S01E01 is held at a stale size (upgrade pending); S02E01 is missing.
        ep_list = [sonarr_ep(1, 1, size=100, release_group="PrivA"), sonarr_ep(2, 1)]

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(44, sd, {"PrivA": [100]}, ep_list)

        assert out["PubA"].urls[self.PUBA_URL].download is True
        assert out["PrivA"].urls[self.PRIVA_URL].download is False
        assert out["PrivB"].urls[self.PRIVB_URL].download is False
        assert hashes == [self.PUBA_HASH]
        # Each SkipNotice reaches the logger at ITS level: the promoted set at
        # INFO, the truly-blocked set at WARNING.
        info = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any("PrivA private-only; grabbing public alternative PubA" in m for m in info), info
        assert any("PrivB private-only (private releases not supported)" in m for m in warnings), warnings
        # The skip flag + group names land on the run context for the grab tail
        # (a promotion succeeded, so no stale hold rides along).
        assert ctx.private_only_skipped is True
        assert ctx.private_only_groups == ["PrivB"]
        assert ctx.private_only_stale_held is False


class TestMixedGroupKeeperPreference:
    """A mixed group's inflated coverage never shadows a fully-public batch.

    The coverage-aware keep gives a mixed group (public S1-only + surviving
    private S1+S2 batch) the same same-files key as a fully-public S1+S2 batch.
    The keeper must prefer the batch whose flagged urls are all addable - a
    mixed keeper's private url is refused at add time, losing S2 - and degrade
    to the add-time WARNING only when no fully-addable group exists.
    """

    H_URL = "https://nyaa.si/view/2"
    H_HASH = "b" * 40

    def _mixed_group_torrents(self) -> tuple[TorrentRecord, TorrentRecord]:
        g_pub = make_torrent_record(
            release_group="G",
            tracker=Tracker.NYAA,
            url=PUB_URL,
            infohash=PUB_HASH,
            file_names=("G.S01E01.web.mkv",),
            file_size=555,
            is_best=True,
        )
        g_priv = make_torrent_record(
            release_group="G",
            tracker=Tracker.ANIMEBYTES,
            url=PRIV_URL,
            infohash=None,
            file_names=("G - S01E01.mkv", "G - S02E01.mkv"),
            file_size=999,
            is_best=True,
        )
        return g_pub, g_priv

    def test_mixed_group_first_still_grabs_the_public_batch(self) -> None:
        # G is first in dict order, but its flagged private url would be refused
        # at add time, so H wins keeper and S2 is actually obtained - pre-fix the
        # keeper was order-dependent and a G win silently lost S2.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            cache_store=cache,
            ctx=ctx,
        )
        g_pub, g_priv = self._mixed_group_torrents()
        h_pub = make_torrent_record(
            release_group="H",
            tracker=Tracker.NYAA,
            url=self.H_URL,
            infohash=self.H_HASH,
            file_names=("H.S01E01.mkv", "H.S02E01.mkv"),
            file_size=777,
            is_best=True,
        )
        entry = make_entry_record(anilist_id=55, torrents=(g_pub, g_priv, h_pub))
        sd = filt.build(entry)
        # The coverage-aware drop keeps G's uncovered private batch.
        assert set(sd["G"].urls) == {PUB_URL, PRIV_URL}
        _fill_episodes(
            sd,
            {
                PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
                PRIV_URL: [
                    EpisodeRecord(season=1, episode=1, size=999),
                    EpisodeRecord(season=2, episode=1, size=999),
                ],
                self.H_URL: [
                    EpisodeRecord(season=1, episode=1, size=777),
                    EpisodeRecord(season=2, episode=1, size=777),
                ],
            },
        )

        # Sonarr is missing both episodes: every url flags before the reducer.
        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(55, sd, {}, [sonarr_ep(1, 1), sonarr_ep(2, 1)])

        assert out["H"].urls[self.H_URL].download is True
        assert out["G"].urls[PUB_URL].download is False
        assert out["G"].urls[PRIV_URL].download is False
        assert hashes == [self.H_HASH]
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.private_only_skipped is False

        torrents = FakeTorrents({self.H_HASH: (AddOutcome.ADDED, "H S01+S02")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="warn",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({55: {}})
        stopped = pipe.grab_and_cache(_grab_request(55, out, hashes, entry))

        assert stopped is False
        assert torrents.calls == [self.H_HASH]
        assert ctx.stats.needs_action == []
        assert cache.check_al_id_in_cache(Arr.SONARR, 55, entry) is True
        assert cache.torrent_hashes(Arr.SONARR, 55) == [self.H_HASH]

    def test_no_public_batch_boundary_warns_at_add_time(self) -> None:
        # No fully-addable group exists: the keeper degrades to the mixed group,
        # its public S1 url is grabbed, and the surviving private batch is
        # refused at add time with a WARNING + the hold flag - never silently.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            cache_store=cache,
            ctx=ctx,
        )
        g_pub, g_priv = self._mixed_group_torrents()
        entry = make_entry_record(anilist_id=56, torrents=(g_pub, g_priv))
        sd = filt.build(entry)
        _fill_episodes(
            sd,
            {
                PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
                PRIV_URL: [
                    EpisodeRecord(season=1, episode=1, size=999),
                    EpisodeRecord(season=2, episode=1, size=999),
                ],
            },
        )

        hashes, out = filt.filter_downloads(56, sd, {}, [sonarr_ep(1, 1), sonarr_ep(2, 1)])
        assert out["G"].urls[PUB_URL].download is True
        assert out["G"].urls[PRIV_URL].download is True
        assert hashes == [PUB_HASH]

        torrents = FakeTorrents({PUB_HASH: (AddOutcome.ADDED, "G S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="warn",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({56: {}})
        with _capture(pipe.log_fmt.logger) as handler:
            pipe.grab_and_cache(_grab_request(56, out, hashes, entry))

        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any("private-only" in m for m in warnings), warnings
        assert ctx.private_only_skipped is True
        assert ctx.private_only_groups == ["G"]
        assert torrents.calls == [PUB_HASH]
        assert cache.check_al_id_in_cache(Arr.SONARR, 56, entry) is True


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
            planner=make_planner(),
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
        assert any("private-only" in m for m in warnings), warnings
        assert ctx.private_only_skipped is True
        assert ctx.private_only_groups == ["A"]
        assert torrents.calls == [PUB_HASH]
        # A non-interactive fallback hold blocks the cache even on a partial grab
        # (the S2 files stay missing), and the no-fallback row resurfaces the title
        # in every run's summary until a covering public release lands.
        assert cache.check_al_id_in_cache(Arr.SONARR, 33, entry) is False
        assert [n.kind for n in ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK]
        # Exact wording pinned once, in test_grab_pipeline's
        # test_private_only_in_fallback_mode_surfaces_no_alternative.
        assert "no public alternative" in ctx.stats.needs_action[0].reason


class TestPromotionGeneralization:
    """The size-mismatch promotion reaches preferred publics (never fallbacks)."""

    def _preferred_pair_entry(self, al_id: int) -> EntryRecord:
        # Both releases are PREFERRED (is_best) with identical coverage - the
        # public one is not a fallback, which the old promotion arm required.
        priv = make_torrent_record(
            release_group="Priv",
            tracker=Tracker.ANIMEBYTES,
            url=PRIV_URL,
            infohash=None,
            file_names=("Show - S01E01.mkv",),
            file_size=999,
            is_best=True,
        )
        pub = make_torrent_record(
            release_group="Pub",
            tracker=Tracker.NYAA,
            url=PUB_URL,
            infohash=PUB_HASH,
            file_names=("Show.S01E01.web.mkv",),
            file_size=555,
            is_best=True,
        )
        return make_entry_record(anilist_id=al_id, torrents=(priv, pub))

    def test_preferred_public_alternative_is_promoted(self) -> None:
        # Upgrade-pending private pick + preferred public twin: pre-graft this
        # warned "no public alternative" and held the title forever; now the
        # public group is promoted, grabbed, and the title caches normally.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            cache_store=cache,
            ctx=ctx,
        )
        entry = self._preferred_pair_entry(66)
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
            hashes, out = filt.filter_downloads(66, sd, {"Priv": [100]}, ep_list)

        assert out["Pub"].urls[PUB_URL].download is True
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == [PUB_HASH]
        info = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
        assert any("grabbing public alternative Pub" in m for m in info), info
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.private_only_skipped is False

        torrents = FakeTorrents({PUB_HASH: (AddOutcome.ADDED, "Show S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({66: {}})
        stopped = pipe.grab_and_cache(_grab_request(66, out, hashes, entry))

        assert stopped is False
        assert torrents.calls == [PUB_HASH]
        assert ctx.stats.needs_action == []
        # Nothing is held, so the fallback-hold uncache must not fire.
        assert cache.check_al_id_in_cache(Arr.SONARR, 66, entry) is True
        cached = cache.get_entry(Arr.SONARR, 66)
        assert cached is not None
        # A promoted PREFERRED public group is not a fallback: no marker.
        assert cached.fallback_satisfied is False

    def test_radarr_preferred_public_alternative_is_promoted(self) -> None:
        # The no-parse (Radarr) twin of the preferred-pair upgrade: the public
        # movie is grabbed instead of the old warn-and-hold.
        ctx = RunContext(arr=Arr.RADARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            ctx=ctx,
        )
        sd = filt.build(self._preferred_pair_entry(73))

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(73, sd, {"Priv": [100]}, None)

        assert out["Pub"].urls[PUB_URL].download is True
        assert out["Priv"].urls[PRIV_URL].download is False
        assert hashes == [PUB_HASH]
        assert not [r for r in handler.records if r.levelno >= logging.WARNING]
        assert ctx.private_only_skipped is False

    M_PUB_URL = "https://nyaa.si/view/3"
    M_PUB_HASH = "c" * 40
    F_URL = "https://nyaa.si/view/4"
    F_HASH = "d" * 40

    def test_owned_mixed_group_holds_its_stale_batch(self) -> None:
        # Mixed group M: its public S1 url is OWNED (unflagged), its private
        # batch holds S2 at a stale size. The only alternative is a FALLBACK,
        # which never replaces the owned stale copy: nothing is promoted, the
        # stale bit is set (no planner notice - the add-time gate warns), and
        # the batch stays flagged for that gate to refuse.
        ctx = RunContext(arr=Arr.SONARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            ctx=ctx,
        )
        m_pub = make_torrent_record(
            release_group="M",
            tracker=Tracker.NYAA,
            url=self.M_PUB_URL,
            infohash=self.M_PUB_HASH,
            file_names=("M.S01E01.web.mkv",),
            file_size=555,
            is_best=True,
        )
        m_priv = make_torrent_record(
            release_group="M",
            tracker=Tracker.ANIMEBYTES,
            url=PRIV_URL,
            infohash=None,
            file_names=("M - S01E01.mkv", "M - S02E01.mkv"),
            file_size=999,
            is_best=True,
        )
        fall = make_torrent_record(
            release_group="F",
            tracker=Tracker.NYAA,
            url=self.F_URL,
            infohash=self.F_HASH,
            file_names=("F.S01E01.mkv", "F.S02E01.mkv"),
            file_size=777,
            is_best=False,
        )
        entry = make_entry_record(anilist_id=77, torrents=(m_pub, m_priv, fall))
        sd = filt.build(entry)
        assert set(sd) == {"M", "F"}
        _fill_episodes(
            sd,
            {
                self.M_PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
                PRIV_URL: [
                    EpisodeRecord(season=1, episode=1, size=999),
                    EpisodeRecord(season=2, episode=1, size=999),
                ],
                self.F_URL: [
                    EpisodeRecord(season=1, episode=1, size=777),
                    EpisodeRecord(season=2, episode=1, size=777),
                ],
            },
        )
        # S01E01 owned at matching size; S02E01 held at a stale size.
        ep_list = [
            sonarr_ep(1, 1, size=555, release_group="M"),
            sonarr_ep(2, 1, size=100, release_group="M"),
        ]

        with _capture(filt.logger) as handler:
            hashes, out = filt.filter_downloads(77, sd, {"M": [555, 100]}, ep_list)

        assert out["F"].urls[self.F_URL].download is False
        assert out["M"].urls[self.M_PUB_URL].download is False
        assert out["M"].urls[PRIV_URL].download is True
        assert hashes == []
        assert handler.records == []
        assert ctx.private_only_skipped is False
        assert ctx.private_only_stale_held is True


class TestEqualUnionMixedGroups:
    """Two equal-union mixed groups: no episode's public url may be lost."""

    A_PUB_URL = "https://nyaa.si/view/5"
    A_PUB_HASH = "a" * 40
    A_PRIV_URL = "https://animebytes.tv/torrents.php?id=2"
    B_PUB_URL = "https://nyaa.si/view/6"
    B_PUB_HASH = "b" * 40
    B_PRIV_URL = "https://animebytes.tv/torrents.php?id=3"

    def _torrents(self) -> tuple[TorrentRecord, TorrentRecord, TorrentRecord, TorrentRecord]:
        a_pub = make_torrent_record(
            release_group="A",
            tracker=Tracker.NYAA,
            url=self.A_PUB_URL,
            infohash=self.A_PUB_HASH,
            file_names=("A.S01E01.mkv",),
            file_size=555,
            is_best=True,
        )
        a_priv = make_torrent_record(
            release_group="A",
            tracker=Tracker.ANIMEBYTES,
            url=self.A_PRIV_URL,
            infohash=None,
            file_names=("A - S01E01.mkv", "A - S02E01.mkv"),
            file_size=999,
            is_best=True,
        )
        b_pub = make_torrent_record(
            release_group="B",
            tracker=Tracker.NYAA,
            url=self.B_PUB_URL,
            infohash=self.B_PUB_HASH,
            file_names=("B.S02E01.mkv",),
            file_size=556,
            is_best=True,
        )
        b_priv = make_torrent_record(
            release_group="B",
            tracker=Tracker.ANIMEBYTES,
            url=self.B_PRIV_URL,
            infohash=None,
            file_names=("B - S01E01.mkv", "B - S02E01.mkv"),
            file_size=998,
            is_best=True,
        )
        return a_pub, a_priv, b_pub, b_priv

    def test_no_episode_lost_regardless_of_group_order(self) -> None:
        # Pre-rescue, the losing group's public url was unflagged wholesale and
        # its episode silently lost (order-dependent; warn mode then cached the
        # loss as done). Both public urls must now grab in either order.
        a_pub, a_priv, b_pub, b_priv = self._torrents()
        for al_id, order in ((91, (a_pub, a_priv, b_pub, b_priv)), (92, (b_pub, b_priv, a_pub, a_priv))):
            ctx = RunContext(arr=Arr.SONARR)
            cache = FakeCacheStore()
            filt = make_release_filter(
                private_releases="warn",
                want_best=True,
                prefer_dual_audio=False,
                planner=make_planner(),
                cache_store=cache,
                ctx=ctx,
            )
            entry = make_entry_record(anilist_id=al_id, torrents=order)
            sd = filt.build(entry)
            _fill_episodes(
                sd,
                {
                    self.A_PUB_URL: [EpisodeRecord(season=1, episode=1, size=555)],
                    self.A_PRIV_URL: [
                        EpisodeRecord(season=1, episode=1, size=999),
                        EpisodeRecord(season=2, episode=1, size=999),
                    ],
                    self.B_PUB_URL: [EpisodeRecord(season=2, episode=1, size=556)],
                    self.B_PRIV_URL: [
                        EpisodeRecord(season=1, episode=1, size=998),
                        EpisodeRecord(season=2, episode=1, size=998),
                    ],
                },
            )

            # Sonarr is missing both episodes: every url flags before the reducer.
            hashes, out = filt.filter_downloads(al_id, sd, {}, [sonarr_ep(1, 1), sonarr_ep(2, 1)])
            assert out["A"].urls[self.A_PUB_URL].download is True, f"order={al_id}"
            assert out["B"].urls[self.B_PUB_URL].download is True, f"order={al_id}"
            assert set(hashes) == {self.A_PUB_HASH, self.B_PUB_HASH}, f"order={al_id}"

            torrents = FakeTorrents(
                {
                    self.A_PUB_HASH: (AddOutcome.ADDED, "A S01"),
                    self.B_PUB_HASH: (AddOutcome.ADDED, "B S02"),
                },
            )
            pipe = make_grab_pipeline(
                cache_store=cache,
                _ctx=ctx,
                _torrents=torrents,
                private_releases="warn",
                sleep_time=0,
            )
            pipe._anilist.al_cache.update({al_id: {}})
            with _capture(pipe.log_fmt.logger) as handler:
                pipe.grab_and_cache(_grab_request(al_id, out, hashes, entry))

            # Both episodes obtained; the keeper's surviving private batch is
            # refused with a WARNING, and warn mode still caches the title.
            assert set(torrents.calls) == {self.A_PUB_HASH, self.B_PUB_HASH}, f"order={al_id}"
            warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
            assert any("private-only" in m for m in warnings), warnings
            assert ctx.private_only_skipped is True
            assert cache.check_al_id_in_cache(Arr.SONARR, al_id, entry) is True


class _CachedEntryReporter:
    """Just enough reporter surface for ``cached_entry_skip``'s logging tail."""

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        state: EntryState = EntryState.UNCHANGED,
    ) -> bool:
        del ctx, arr, al_id, state
        return True


class TestModeSwitchResurfacesFallbackSatisfied:
    """A fallback-satisfied title resurfaces under warn mode, and only there.

    The marker exists so warn's "warn every run" contract survives a spell in
    fallback mode: without it, the fallback-satisfied title stays cache-skipped
    until SeaDex updates the entry.
    """

    def test_fallback_marker_resurfaces_under_warn_and_persists(self) -> None:
        # 1) fallback run: Sonarr misses the episode, the keeper flow grabs the
        # public fallback, and the title caches with the marker set.
        ctx = RunContext(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
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
        hashes, out = filt.filter_downloads(11, sd, {}, [sonarr_ep(1, 1)])
        assert out["Fall"].urls[PUB_URL].download is True

        torrents = FakeTorrents({PUB_HASH: (AddOutcome.ADDED, "Fall S01")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _ctx=ctx,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe._anilist.al_cache.update({11: {}})
        pipe.grab_and_cache(_grab_request(11, out, hashes, entry))
        cached = cache.get_entry(Arr.SONARR, 11)
        assert cached is not None
        assert cached.fallback_satisfied is True

        # 2) warn services over the same cache: BOTH gates re-process the id
        # (prefetch warming and the loop's skip must agree).
        seadex = FakeSeaDexSource({11: entry})
        warn_run = make_services(
            cache_store=cache,
            _seadex=seadex,
            _reporter=_CachedEntryReporter(),
            private_releases="warn",
        )
        assert warn_run.al_id_needs_scan(11) is True
        assert warn_run.cached_entry_skip(11, entry, lambda: "") is False

        # 3) the warn-mode reprocess warns and holds: the private pick is all
        # that's offered (the Arr owns the grabbed fallback's file), nothing is
        # cached, and the marker persists for the next run.
        warn_ctx = RunContext(arr=Arr.SONARR)
        warn_filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            cache_store=cache,
            ctx=warn_ctx,
        )
        sd_warn = warn_filt.build(entry)
        assert set(sd_warn) == {"Priv"}  # warn mode adds no fallback
        _fill_episodes(sd_warn, {PRIV_URL: [EpisodeRecord(season=1, episode=1, size=999)]})
        ep_list = [sonarr_ep(1, 1, size=555, release_group="Fall")]
        with _capture(warn_filt.logger) as handler:
            warn_hashes, warn_out = warn_filt.filter_downloads(11, sd_warn, {"Fall": [555]}, ep_list)
        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert any("private-only (private releases not supported)" in m for m in warnings), warnings

        warn_pipe = make_grab_pipeline(cache_store=cache, _ctx=warn_ctx, private_releases="warn", sleep_time=0)
        warn_pipe.grab_and_cache(_grab_request(11, warn_out, warn_hashes, entry))
        assert [n.kind for n in warn_ctx.stats.needs_action] == [NeedsActionKind.PRIVATE_ONLY]
        persisted = cache.get_entry(Arr.SONARR, 11)
        assert persisted is not None
        assert persisted.fallback_satisfied is True  # the hold wrote nothing

        # 4) switching back to fallback mode: the marked entry skips as cached.
        fallback_run = make_services(
            cache_store=cache,
            _seadex=seadex,
            _reporter=_CachedEntryReporter(),
            private_releases="fallback",
        )
        assert fallback_run.al_id_needs_scan(11) is False
        assert fallback_run.cached_entry_skip(11, entry, lambda: "") is True
