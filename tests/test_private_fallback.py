# pyright: strict
"""End-to-end regressions for `seadex.private_releases: fallback`.

Drive the full build -> plan_grabs -> grab_and_cache path over one SeaDex
entry, pinning the fallback contract: when a preferred release is private-only,
grab the entry's best public fallback. Warn when no public fallback
exists, or when the Arr owns the preferred private release at an outdated size (a
fallback never replaces an owned copy). Regression coverage for the
preferred-public size-mismatch promote and the coverage-blind per-group drop.
"""

from seadex import EntryRecord, TorrentRecord, Tracker

from pearlarr.arr_models import ArrFiles
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.output import NeedsActionCause, ReleaseSkipped, Severity, TitleDetail
from pearlarr.releases import PlacedEpisode, SeadexDict, SkipReason

from .builders import (
    AddStatus,
    FakeCacheStore,
    FakeSeaDexSource,
    FakeTorrents,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release_filter,
    make_services,
    make_torrent_record,
    placed_episode,
    sonarr_ep,
    title_is_cached,
)
from .fakes import RecordingHub, install_recording_hub

PRIV_URL = "https://animebytes.tv/torrents.php?id=1"
PUB_URL = "https://nyaa.si/view/1"
PUB_HASH = "f" * 40

# The outdated-copy private-only skip's planner notice and needs-action row wording, pinned.
OUTDATED_NOTICE = (
    "private-only; your copy is outdated (its file size no longer matches the release) and only a fallback covers it"
)
OUTDATED_ROW_REASON = (
    "private-only release; your copy is outdated (its file size no longer matches) and only a fallback covers it"
)


def _entry_private_preferred_plus_public_alt() -> EntryRecord:
    """A SeaDex entry preferring a private-only release, with a public fallback."""

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


def _attach_episodes(sd: SeadexDict, mapping: dict[str, list[PlacedEpisode]]) -> None:
    """Mirror `attach_placements`: per-release episodes + the group union."""

    for group_releases in sd.values():
        all_eps: list[PlacedEpisode] = []
        group_releases.episodes = all_eps
        for url, item in group_releases.releases.items():
            eps = list(mapping.get(url, []))
            item.episodes = eps
            all_eps.extend(eps)


def _warning_texts(recording: RecordingHub) -> list[str]:
    """The recorded detail texts at WARNING+ (the old `levelno >= WARNING` filter)."""

    return [d.value.text for d in recording.of_type(TitleDetail) if d.severity >= Severity.WARNING]


def _info_texts(recording: RecordingHub) -> list[str]:
    """The recorded detail texts at exactly INFO."""

    return [d.value.text for d in recording.of_type(TitleDetail) if d.severity is Severity.INFO]


def _private_skips(recording: RecordingHub) -> list[ReleaseSkipped]:
    """The add-time private-only skips (grab_pipeline's typed `ReleaseSkipped` facts)."""

    return [s for s in recording.of_type(ReleaseSkipped) if s.reason is SkipReason.PRIVATE_ONLY]


def _grab_request(anilist_id: int, seadex_dict: SeadexDict, entry: EntryRecord) -> GrabRequest:
    """The per-title grab payload as the strategies assemble it."""

    return GrabRequest(
        anilist_id=anilist_id,
        arr_title="Show",
        title_name="Show",
        entry=entry,
        seadex_dict=seadex_dict,
        update={"name": "Show", "updated_at": entry.updated_at},
        replaced_groups=(),
    )


class TestUpgradePendingSkipsForOwnedOutdated:
    """An owned private release at an outdated size warns and is skipped - a fallback never replaces it."""

    def test_sonarr_upgrade_pending_warns_and_skips(self) -> None:
        # The Arr has Priv's release at an OUTDATED size (the SeaDex entry updated):
        # a fallback (a cascade loser) must not supersede the owned preferred
        # copy, so nothing is promoted and the set warns with the outdated reason.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = _entry_private_preferred_plus_public_alt()
        sd = filt.build(entry)
        _attach_episodes(
            sd,
            {
                PRIV_URL: [placed_episode(season=1, episode=1, size=999)],
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )
        title_episodes = [sonarr_ep(1, 1, size=100, release_group="Priv")]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (100,)}), title_episodes)

        assert plan.seadex_dict["Fall"].releases[PUB_URL].to_grab is False
        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        warnings = _warning_texts(recording)
        assert any(OUTDATED_NOTICE in m for m in warnings), warnings
        assert state.per_title.private_only_skipped is True
        assert state.per_title.private_only_outdated is True

        # The private-only skip keeps the title uncached and surfaces the PRIVATE_ONLY_OUTDATED row,
        # so it resurfaces every run until the copy is updated or deleted.
        pipe = make_grab_pipeline(cache_store=cache, _state=state, private_releases="fallback", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, plan.seadex_dict, entry))
        assert cache.get_cached_title(Arr.SONARR, 11) is None
        assert [n.cause for n in state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_OUTDATED]
        assert [n.reason for n in state.stats.needs_action] == [OUTDATED_ROW_REASON]

    def test_radarr_size_disjoint_warns_and_skips(self) -> None:
        # The no-episode (Radarr) twin: the size-disjoint branch marks the private
        # release to grab again, and the same-set fallback stays un-promoted - a private-only skip.
        state = ArrRunState(arr=Arr.RADARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        sd = filt.build(_entry_private_preferred_plus_public_alt())

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (100,)}), None)

        assert plan.seadex_dict["Fall"].releases[PUB_URL].to_grab is False
        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        warnings = _warning_texts(recording)
        assert any(OUTDATED_NOTICE in m for m in warnings), warnings
        assert state.per_title.private_only_skipped is True
        assert state.per_title.private_only_outdated is True


class TestOwnedPreferredPrivateAtMatchingSize:
    """The Arr owns the preferred private release at SeaDex's size: fully protected.

    The matcher never marks to grab the private release (group + size match) NOR the
    fallback beside it (another preferred group already covers its files), so the title
    reads up-to-date and is cached - no notices, no needs-action rows.
    Also mutation-kills the two matcher already-covered gates.
    """

    def test_sonarr_owned_private_at_matching_size_stays_untouched(self) -> None:
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = _entry_private_preferred_plus_public_alt()
        sd = filt.build(entry)
        _attach_episodes(
            sd,
            {
                PRIV_URL: [placed_episode(season=1, episode=1, size=999)],
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )
        # Sonarr has Priv's release at the MATCHING SeaDex size.
        title_episodes = [sonarr_ep(1, 1, size=999, release_group="Priv")]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (999,)}), title_episodes)

        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        assert plan.seadex_dict["Fall"].releases[PUB_URL].to_grab is False
        assert recording.events == []
        assert state.per_title.private_only_skipped is False
        assert state.per_title.private_only_outdated is False

        pipe = make_grab_pipeline(cache_store=cache, _state=state, private_releases="fallback", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, plan.seadex_dict, entry))
        assert state.stats.up_to_date == 1
        assert state.stats.needs_action == []
        cached = cache.get_cached_title(Arr.SONARR, 11)
        assert title_is_cached(cached, entry) is True
        assert cached is not None
        # The owned PREFERRED private release satisfied the title, not the idle
        # fallback pool - the false positive the coarse any(is_fallback) had.
        assert cached.fallback_satisfied is False

    def test_radarr_owned_private_at_matching_size_stays_untouched(self) -> None:
        # The no-episode (Radarr) twin, via the arr size overlap.
        state = ArrRunState(arr=Arr.RADARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = _entry_private_preferred_plus_public_alt()
        sd = filt.build(entry)

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (999,)}), None)

        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        assert plan.seadex_dict["Fall"].releases[PUB_URL].to_grab is False
        assert recording.events == []
        assert state.per_title.private_only_skipped is False
        assert state.per_title.private_only_outdated is False

        pipe = make_grab_pipeline(cache_store=cache, _state=state, private_releases="fallback", sleep_time=0)
        pipe.grab_and_cache(_grab_request(22, plan.seadex_dict, entry))
        assert state.stats.up_to_date == 1
        assert state.stats.needs_action == []
        cached = cache.get_cached_title(Arr.RADARR, 22)
        assert title_is_cached(cached, entry) is True
        assert cached is not None
        assert cached.fallback_satisfied is False

    def test_warn_mode_upgrade_pending_warns_and_skips(self) -> None:
        # Parity: warn mode on the same upgrade-pending state warns and leaves
        # the title uncached, surfacing the plain PRIVATE_ONLY needs-action row.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = _entry_private_preferred_plus_public_alt()
        sd = filt.build(entry)
        assert set(sd) == {"Priv"}  # warn mode adds no fallback

        _attach_episodes(sd, {PRIV_URL: [placed_episode(season=1, episode=1, size=999)]})
        title_episodes = [sonarr_ep(1, 1, size=100, release_group="Priv")]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (100,)}), title_episodes)

        warnings = _warning_texts(recording)
        assert any("private-only (private releases not supported)" in m for m in warnings), warnings
        assert state.per_title.private_only_skipped is True

        pipe = make_grab_pipeline(cache_store=cache, _state=state, private_releases="warn", sleep_time=0)
        pipe.grab_and_cache(_grab_request(11, plan.seadex_dict, entry))

        assert cache.get_cached_title(Arr.SONARR, 11) is None
        assert [n.cause for n in state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY]


class TestPlanGrabsNotice:
    """plan_grabs renders each PrivateOnlyNotice at its level and stamps the state."""

    PRIVA_URL = "https://animebytes.tv/torrents.php?id=10"
    PRIVB_URL = "https://animebytes.tv/torrents.php?id=20"
    PUBA_URL = "https://nyaa.si/view/9"
    PUBA_HASH = "e" * 40

    def test_renders_both_levels_and_carries_skip_state(self) -> None:
        # One entry, two same-files sets: {PrivA, PubA} promotes the preferred
        # public group (an INFO notice). {PrivB} is private-only with no cover
        # (a WARNING notice, plus the skip flag + group carried onto the
        # ArrRunState).
        state = ArrRunState(arr=Arr.SONARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
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
        _attach_episodes(
            sd,
            {
                self.PRIVA_URL: [placed_episode(season=1, episode=1, size=999)],
                self.PRIVB_URL: [placed_episode(season=2, episode=1, size=777)],
                self.PUBA_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )
        # S01E01 has a file at an outdated size (upgrade pending). S02E01 is missing.
        title_episodes = [sonarr_ep(1, 1, size=100, release_group="PrivA"), sonarr_ep(2, 1)]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"PrivA": (100,)}), title_episodes)

        assert plan.seadex_dict["PubA"].releases[self.PUBA_URL].to_grab is True
        assert plan.seadex_dict["PrivA"].releases[self.PRIVA_URL].to_grab is False
        assert plan.seadex_dict["PrivB"].releases[self.PRIVB_URL].to_grab is False
        # Each PrivateOnlyNotice reaches the logger at ITS level: the promoted set at
        # INFO, the truly-blocked set at WARNING.
        info = _info_texts(recording)
        warnings = _warning_texts(recording)
        assert any("PrivA private-only; grabbing public release PubA instead" in m for m in info), info
        assert any("PrivB private-only (private releases not supported)" in m for m in warnings), warnings
        # The skip flag + group names land on the arr run state for the grab tail
        # (a promotion succeeded, so the outdated bit stays unset).
        assert state.per_title.private_only_skipped is True
        assert state.per_title.private_only_groups == ["PrivB"]
        assert state.per_title.private_only_outdated is False


class TestMixedGroupKeeperPreference:
    """A mixed group's inflated coverage never shadows a fully-public group.

    The coverage-aware keep gives a mixed group (public S1-only + surviving
    private S1+S2 torrent) the same same-files key as a fully-public S1+S2 group.
    The keeper must prefer the group whose releases to grab are all addable - a
    mixed keeper's private release is skipped at add time, losing S2 - and degrade
    to the add-time WARNING only when no fully-addable group exists.
    """

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

    def test_no_public_batch_boundary_warns_at_add_time(self) -> None:
        # No fully-addable group exists: the keeper degrades to the mixed group,
        # its public S1 release is grabbed, and the surviving private torrent is
        # skipped at add time with a WARNING + the private-only skip flag - never silently.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        g_pub, g_priv = self._mixed_group_torrents()
        entry = make_entry_record(anilist_id=56, torrents=(g_pub, g_priv))
        sd = filt.build(entry)
        _attach_episodes(
            sd,
            {
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
                PRIV_URL: [
                    placed_episode(season=1, episode=1, size=999),
                    placed_episode(season=2, episode=1, size=999),
                ],
            },
        )

        plan = filt.plan_grabs(sd, ArrFiles(), [sonarr_ep(1, 1), sonarr_ep(2, 1)])
        assert plan.seadex_dict["G"].releases[PUB_URL].to_grab is True
        assert plan.seadex_dict["G"].releases[PRIV_URL].to_grab is True

        torrents = FakeTorrents({PUB_HASH: (AddStatus.ADDED, "G S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _state=state,
            _torrents=torrents,
            private_releases="warn",
            sleep_time=0,
        )
        recording = install_recording_hub()
        pipe.grab_and_cache(_grab_request(56, plan.seadex_dict, entry))

        assert [s.group for s in _private_skips(recording)] == ["G"]
        assert state.per_title.private_only_skipped is True
        assert state.per_title.private_only_groups == ["G"]
        assert torrents.calls == [PUB_HASH]
        assert title_is_cached(cache.get_cached_title(Arr.SONARR, 56), entry) is True


class TestSurvivingPrivateCoverage:
    """A private release kept for uncovered files is warned about, never silently lost."""

    def test_surviving_private_batch_warns_at_add_time(self) -> None:
        # One group: a private S1+S2 torrent survives the coverage-aware drop next
        # to its public S1-only fallback. With both episodes missing, the torrent
        # stays marked to grab and the add-time gate warns + sets the private-only skip flag - the
        # old drop lost the S2 coverage without a trace.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=True,
            planner=make_planner(),
            state=state,
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
        assert set(sd["A"].releases) == {PRIV_URL, PUB_URL}

        _attach_episodes(
            sd,
            {
                PRIV_URL: [
                    placed_episode(season=1, episode=1, size=999),
                    placed_episode(season=2, episode=1, size=999),
                ],
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )

        # Sonarr is missing BOTH episodes: the torrent and the fallback are both marked to grab.
        plan = filt.plan_grabs(sd, ArrFiles(), [sonarr_ep(1, 1), sonarr_ep(2, 1)])
        assert plan.seadex_dict["A"].releases[PRIV_URL].to_grab is True
        assert plan.seadex_dict["A"].releases[PUB_URL].to_grab is True

        torrents = FakeTorrents({PUB_HASH: (AddStatus.ADDED, "A S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _state=state,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        recording = install_recording_hub()
        pipe.grab_and_cache(_grab_request(33, plan.seadex_dict, entry))

        assert [s.group for s in _private_skips(recording)] == ["A"]
        assert state.per_title.private_only_skipped is True
        assert state.per_title.private_only_groups == ["A"]
        assert torrents.calls == [PUB_HASH]
        # A non-interactive private-only skip in fallback mode blocks the cache even on a partial grab
        # (the S2 files stay missing), and the no-fallback row resurfaces the title
        # in every run's summary until a covering public release lands.
        assert title_is_cached(cache.get_cached_title(Arr.SONARR, 33), entry) is False
        assert [n.cause for n in state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK]
        # Exact wording pinned once, in test_grab_pipeline's
        # test_private_only_in_fallback_mode_surfaces_no_alternative.
        assert "no fallback" in state.stats.needs_action[0].reason


class TestPromotionGeneralization:
    """The size-mismatch promotion reaches preferred publics (never fallbacks)."""

    def _preferred_pair_entry(self, anilist_id: int) -> EntryRecord:
        # Both releases are PREFERRED (is_best) with identical coverage - the
        # public one is not a fallback, which the old promotion branch required.
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
        return make_entry_record(anilist_id=anilist_id, torrents=(priv, pub))

    def test_preferred_public_alternative_is_promoted(self) -> None:
        # Upgrade-pending private release + preferred public twin: pre-graft this
        # warned "no public alternative" and kept the title uncached forever. Now the
        # public group is promoted, grabbed, and the title caches normally.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = self._preferred_pair_entry(66)
        sd = filt.build(entry)
        _attach_episodes(
            sd,
            {
                PRIV_URL: [placed_episode(season=1, episode=1, size=999)],
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )
        title_episodes = [sonarr_ep(1, 1, size=100, release_group="Priv")]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (100,)}), title_episodes)

        assert plan.seadex_dict["Pub"].releases[PUB_URL].to_grab is True
        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        info = _info_texts(recording)
        assert any("grabbing public release Pub instead" in m for m in info), info
        assert _warning_texts(recording) == []
        assert state.per_title.private_only_skipped is False

        torrents = FakeTorrents({PUB_HASH: (AddStatus.ADDED, "Show S01 web")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _state=state,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe.grab_and_cache(_grab_request(66, plan.seadex_dict, entry))
        assert torrents.calls == [PUB_HASH]
        assert state.stats.needs_action == []
        # Nothing is skipped, so the private-only skip's uncache must not fire.
        cached = cache.get_cached_title(Arr.SONARR, 66)
        assert title_is_cached(cached, entry) is True
        assert cached is not None
        # A promoted PREFERRED public group is not a fallback: no marker.
        assert cached.fallback_satisfied is False

    def test_radarr_preferred_public_alternative_is_promoted(self) -> None:
        # The no-parse (Radarr) twin of the preferred-pair upgrade: the public
        # movie is grabbed instead of the old private-only skip.
        state = ArrRunState(arr=Arr.RADARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        sd = filt.build(self._preferred_pair_entry(73))

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"Priv": (100,)}), None)

        assert plan.seadex_dict["Pub"].releases[PUB_URL].to_grab is True
        assert plan.seadex_dict["Priv"].releases[PRIV_URL].to_grab is False
        assert _warning_texts(recording) == []
        assert state.per_title.private_only_skipped is False

    M_PUB_URL = "https://nyaa.si/view/3"
    M_PUB_HASH = "c" * 40
    F_URL = "https://nyaa.si/view/4"
    F_HASH = "d" * 40

    def test_owned_mixed_group_skips_its_outdated_torrent(self) -> None:
        # Mixed group M: its public S1 release is OWNED (not marked to grab), its private
        # torrent's S2 is owned at an outdated size. The only other release is a FALLBACK,
        # which never replaces the owned outdated copy: nothing is promoted, the
        # outdated bit is set (no planner notice - the add-time gate warns), and
        # the torrent stays marked to grab for that gate to skip.
        state = ArrRunState(arr=Arr.SONARR)
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
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
        _attach_episodes(
            sd,
            {
                self.M_PUB_URL: [placed_episode(season=1, episode=1, size=555)],
                PRIV_URL: [
                    placed_episode(season=1, episode=1, size=999),
                    placed_episode(season=2, episode=1, size=999),
                ],
                self.F_URL: [
                    placed_episode(season=1, episode=1, size=777),
                    placed_episode(season=2, episode=1, size=777),
                ],
            },
        )
        # S01E01 owned at matching size. S02E01 owned at an outdated size.
        title_episodes = [
            sonarr_ep(1, 1, size=555, release_group="M"),
            sonarr_ep(2, 1, size=100, release_group="M"),
        ]

        recording = install_recording_hub()
        plan = filt.plan_grabs(sd, ArrFiles(by_group={"M": (555, 100)}), title_episodes)

        assert plan.seadex_dict["F"].releases[self.F_URL].to_grab is False
        assert plan.seadex_dict["M"].releases[self.M_PUB_URL].to_grab is False
        assert plan.seadex_dict["M"].releases[PRIV_URL].to_grab is True
        assert recording.events == []
        assert state.per_title.private_only_skipped is False
        assert state.per_title.private_only_outdated is True


class _CachedTitleReporter:
    """Just enough reporter surface for `is_unchanged`'s logging tail."""

    def log_cached_title(self, state: ArrRunState, anilist_id: int) -> None:
        del state, anilist_id


class TestModeSwitchResurfacesFallbackSatisfied:
    """A fallback-satisfied title resurfaces under warn mode, and only there.

    The marker exists so warn's "warn every run" contract survives a spell in
    fallback mode: without it, the fallback-satisfied title stays cache-skipped
    until SeaDex updates the entry.
    """

    def test_fallback_marker_resurfaces_under_warn_and_persists(self) -> None:
        # 1) fallback run: Sonarr misses the episode, the keeper flow grabs the
        # public fallback, and the title caches with the marker set.
        state = ArrRunState(arr=Arr.SONARR)
        cache = FakeCacheStore()
        filt = make_release_filter(
            private_releases="fallback",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=state,
        )
        entry = _entry_private_preferred_plus_public_alt()
        sd = filt.build(entry)
        _attach_episodes(
            sd,
            {
                PRIV_URL: [placed_episode(season=1, episode=1, size=999)],
                PUB_URL: [placed_episode(season=1, episode=1, size=555)],
            },
        )
        plan = filt.plan_grabs(sd, ArrFiles(), [sonarr_ep(1, 1)])
        assert plan.seadex_dict["Fall"].releases[PUB_URL].to_grab is True

        torrents = FakeTorrents({PUB_HASH: (AddStatus.ADDED, "Fall S01")})
        pipe = make_grab_pipeline(
            cache_store=cache,
            _state=state,
            _torrents=torrents,
            private_releases="fallback",
            sleep_time=0,
        )
        pipe.grab_and_cache(_grab_request(11, plan.seadex_dict, entry))
        cached = cache.get_cached_title(Arr.SONARR, 11)
        assert cached is not None
        assert cached.fallback_satisfied is True

        # 2) a later fallback-mode run skips the marked title as cached: the marker
        # re-opens nothing there.
        seadex = FakeSeaDexSource({11: entry})
        fallback_run = make_services(
            cache_store=cache,
            _seadex=seadex,
            _reporter=_CachedTitleReporter(),
            private_releases="fallback",
        )
        assert fallback_run.title_needs_check(11) is False
        assert fallback_run.is_unchanged(11, entry, lambda: "") is True

        # 3) warn services over the same cache: BOTH gates re-process the title
        # (prefetch warming and the loop's skip must agree).
        warn_run = make_services(
            cache_store=cache,
            _seadex=seadex,
            _reporter=_CachedTitleReporter(),
            private_releases="warn",
        )
        assert warn_run.title_needs_check(11) is True
        assert warn_run.is_unchanged(11, entry, lambda: "") is False

        # 4) the warn-mode reprocess warns and skips: the private release is all
        # that's offered (the Arr owns the grabbed fallback's file), nothing is
        # cached, and the marker persists for the next run.
        warn_state = ArrRunState(arr=Arr.SONARR)
        warn_filt = make_release_filter(
            private_releases="warn",
            want_best=True,
            prefer_dual_audio=False,
            planner=make_planner(),
            state=warn_state,
        )
        sd_warn = warn_filt.build(entry)
        assert set(sd_warn) == {"Priv"}  # warn mode adds no fallback
        _attach_episodes(sd_warn, {PRIV_URL: [placed_episode(season=1, episode=1, size=999)]})
        title_episodes = [sonarr_ep(1, 1, size=555, release_group="Fall")]
        recording = install_recording_hub()
        warn_plan = warn_filt.plan_grabs(sd_warn, ArrFiles(by_group={"Fall": (555,)}), title_episodes)
        warnings = _warning_texts(recording)
        assert any("private-only (private releases not supported)" in m for m in warnings), warnings

        warn_pipe = make_grab_pipeline(cache_store=cache, _state=warn_state, private_releases="warn", sleep_time=0)
        warn_pipe.grab_and_cache(_grab_request(11, warn_plan.seadex_dict, entry))
        assert [n.cause for n in warn_state.stats.needs_action] == [NeedsActionCause.PRIVATE_ONLY]
        persisted = cache.get_cached_title(Arr.SONARR, 11)
        assert persisted is not None
        assert persisted.fallback_satisfied is True  # the skip left the marker alone
