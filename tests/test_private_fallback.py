# pyright: strict
"""End-to-end regressions for `seadex.private_releases: fallback`.

Drive the full build -> plan_grabs -> grab_and_cache path over one SeaDex
entry, pinning the fallback contract: when a preferred release is private-only,
grab the entry's best public fallback. Warn when no public fallback
exists, or when the Arr owns the preferred private release at an outdated size (a
fallback never replaces an owned copy). Regression coverage for the
preferred-public size-mismatch promote and the coverage-blind per-group drop.
"""

import re
from dataclasses import dataclass

import pytest
from seadex import EntryRecord, TorrentRecord, Tracker

from pearlarr.arr_models import ArrFiles, SonarrEpisode
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
    arr_files,
    grab_request,
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

# What the plan's notices and the needs-action rows say of a private-only skip.
PLAIN_NOTICE = "private-only (private releases not supported)"
OUTDATED_NOTICE = (
    "private-only; your copy is outdated (its file size no longer matches the release) and only a fallback covers it"
)
PLAIN_ROW = "private-only release; private releases not supported"
NO_FALLBACK_ROW = "private-only release; no fallback covers these files"
OUTDATED_ROW = (
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _Torrent:
    """A torrent SeaDex lists for a release of `group`: one file per episode, each `size` bytes."""

    group: str
    public: bool
    episodes: tuple[str, ...]
    size: int
    preferred: bool


def _private(group: str, *episodes: str, size: int) -> _Torrent:
    """A private tracker's torrent of `group`'s release of `episodes`, such as "S01E01", which SeaDex prefers."""

    return _Torrent(group=group, public=False, episodes=episodes, size=size, preferred=True)


def _public(group: str, *episodes: str, size: int, preferred: bool = True) -> _Torrent:
    """A public tracker's torrent of `group`'s release of `episodes`."""

    return _Torrent(group=group, public=True, episodes=episodes, size=size, preferred=preferred)


def _entry(*torrents: _Torrent) -> EntryRecord:
    """A SeaDex entry listing `torrents`: a public one on Nyaa with an infohash, a private one on AnimeBytes with none."""

    records = tuple(
        make_torrent_record(
            release_group=torrent.group,
            tracker=Tracker.NYAA if torrent.public else Tracker.ANIMEBYTES,
            url=f"https://nyaa.si/view/{number}"
            if torrent.public
            else f"https://animebytes.tv/torrents.php?id={number}",
            infohash=f"{number:040x}" if torrent.public else None,
            file_names=tuple(f"{torrent.group} - {episode}.mkv" for episode in torrent.episodes),
            file_size=torrent.size,
            is_best=torrent.preferred,
        )
        for number, torrent in enumerate(torrents, start=1)
    )
    return make_entry_record(torrents=records)


@dataclass(frozen=True, slots=True)
class _Library:
    """What the arr has of the title: a Sonarr title's episodes, or a Radarr movie's files, which have no episodes."""

    arr: Arr
    files: ArrFiles
    episodes: tuple[SonarrEpisode, ...] | None


def _sonarr(*episodes: SonarrEpisode) -> _Library:
    """A Sonarr title of `episodes`, with their files by group as Sonarr reports them."""

    return _Library(Arr.SONARR, arr_files(episodes), episodes)


def _radarr(files: ArrFiles) -> _Library:
    """A Radarr movie with `files`."""

    return _Library(Arr.RADARR, files, None)


def _missing(season: int, episode: int) -> SonarrEpisode:
    """An episode Sonarr has no file for."""

    return sonarr_ep(season, episode, episode_file_id=0)


@dataclass(frozen=True, slots=True)
class _Detail:
    """A line of the title's block: its label, its text, and its level."""

    label: str
    text: str
    severity: Severity = Severity.INFO


@dataclass(frozen=True, slots=True)
class _Skip:
    """A release the grab skipped with a warning when it came to add it: its group, and why."""

    group: str
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class _Row:
    """The title's needs-action row in the summary: the groups it names, why, and the cause its tip is chosen by."""

    group: str
    reason: str
    cause: NeedsActionCause


@dataclass(frozen=True, slots=True, kw_only=True)
class _Outcome:
    """What a run shows of one title's check, and what it stores for later runs."""

    added: tuple[str, ...] = ()
    """The groups whose release qBittorrent added."""
    details: tuple[_Detail, ...] = ()
    """The title's lines: the plan's private-only notices, and its status when it has nothing to grab."""
    skips: tuple[_Skip, ...] = ()
    """Every release the grab skipped when it came to add it."""
    needs_action: tuple[_Row, ...] = ()
    """The title's rows in the run's summary of what needs a human."""
    cached: bool = False
    """Stored: later runs leave a cached title unchecked until SeaDex updates the entry."""
    fallback_marker: bool = False
    """Stored: the title was cached as satisfied by a fallback, which a warn mode run checks again."""


_EPISODE_NUMBERS = re.compile(r"S(\d+)E(\d+)")


def _place_by_name(seadex_dict: SeadexDict) -> None:
    """Place each release's files on the episode each file's name numbers, and each group on its releases' episodes."""

    for group_releases in seadex_dict.values():
        group_episodes: list[PlacedEpisode] = []
        for release in group_releases.releases.values():
            release.episodes = []
            for name, size in zip(release.files, release.file_sizes, strict=True):
                numbers = _EPISODE_NUMBERS.search(name)
                assert numbers is not None, name
                release.episodes.append(placed_episode(int(numbers[1]), int(numbers[2]), size))
            group_episodes.extend(release.episodes)
        group_releases.episodes = group_episodes


def _outcome(entry: EntryRecord, library: _Library, *, private_releases: str) -> _Outcome:
    """The outcome of a run's check of `entry`'s title against `library` under `private_releases`.

    qBittorrent adds every public release asked for.
    """

    cache = FakeCacheStore()
    recording = install_recording_hub()
    state = ArrRunState(arr=library.arr)
    release_filter = make_release_filter(
        private_releases=private_releases, planner=make_planner(arr=library.arr), state=state
    )
    seadex_dict = release_filter.build(entry)
    if library.episodes is not None:
        _place_by_name(seadex_dict)
    plan = release_filter.plan_grabs(seadex_dict, library.files, library.episodes)
    group_by_hash: dict[str | None, str] = {t.infohash: t.release_group for t in entry.torrents if t.infohash}
    torrents = FakeTorrents({infohash: (AddStatus.ADDED, group) for infohash, group in group_by_hash.items()})
    pipeline = make_grab_pipeline(
        cache_store=cache, _state=state, _torrents=torrents, private_releases=private_releases
    )

    pipeline.grab_and_cache(
        grab_request(
            anilist_id=entry.anilist_id,
            entry=entry,
            seadex_dict=plan.seadex_dict,
            update={"updated_at": entry.updated_at},
        )
    )

    row = cache.get_cached_title(library.arr, entry.anilist_id)
    cached = title_is_cached(row, entry)
    return _Outcome(
        added=tuple(group_by_hash[infohash] for infohash in torrents.calls),
        details=tuple(_Detail(d.label, d.value.text, d.severity) for d in recording.of_type(TitleDetail)),
        skips=tuple(_Skip(skip.group, skip.reason) for skip in recording.of_type(ReleaseSkipped)),
        needs_action=tuple(_Row(r.group, r.reason, r.cause) for r in state.stats.needs_action),
        cached=cached,
        fallback_marker=row is not None and cached and row.fallback_satisfied,
    )


class TestTheArrHasACopyOfThePrivateRelease:
    """The arr has a file of the private release SeaDex prefers, at the size SeaDex lists or an outdated one."""

    @pytest.mark.parametrize(
        "library",
        [
            pytest.param(_sonarr(sonarr_ep(1, 1, size=999, release_group="Priv")), id="sonarr"),
            pytest.param(_radarr(ArrFiles(by_group={"Priv": (999,)})), id="radarr"),
        ],
    )
    def test_the_private_release_at_its_listed_size_is_up_to_date(self, library: _Library) -> None:
        # Fall's fallback covers the same file, so it isn't grabbed either, and the title is cached with no fallback
        # marker: the arr's copy of the preferred release satisfies it.
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))

        outcome = _outcome(entry, library, private_releases="fallback")

        assert outcome == _Outcome(details=(_Detail("status", "up to date"),), cached=True)

    @pytest.mark.parametrize(
        "library",
        [
            pytest.param(_sonarr(sonarr_ep(1, 1, size=100, release_group="Priv")), id="sonarr"),
            pytest.param(_radarr(ArrFiles(by_group={"Priv": (100,)})), id="radarr"),
        ],
    )
    def test_a_fallback_never_replaces_an_outdated_copy_so_the_title_warns_every_run(self, library: _Library) -> None:
        # Only the private release would replace the arr's copy at a size SeaDex no longer lists, and it is never
        # grabbed.
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))

        outcome = _outcome(entry, library, private_releases="fallback")

        assert outcome == _Outcome(
            details=(_Detail("skipped", f"Priv {OUTDATED_NOTICE}", Severity.WARNING),),
            needs_action=(_Row("Priv", OUTDATED_ROW, NeedsActionCause.PRIVATE_ONLY_OUTDATED),),
        )

    def test_warn_mode_gives_the_plain_warning_for_an_outdated_copy(self) -> None:
        # Warn mode offers no fallback, so the warning and the row are the plain ones.
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))

        outcome = _outcome(entry, _sonarr(sonarr_ep(1, 1, size=100, release_group="Priv")), private_releases="warn")

        assert outcome == _Outcome(
            details=(_Detail("skipped", f"Priv {PLAIN_NOTICE}", Severity.WARNING),),
            needs_action=(_Row("Priv", PLAIN_ROW, NeedsActionCause.PRIVATE_ONLY),),
        )

    @pytest.mark.parametrize(
        "library",
        [
            pytest.param(_sonarr(sonarr_ep(1, 1, size=100, release_group="Priv")), id="sonarr"),
            pytest.param(_radarr(ArrFiles(by_group={"Priv": (100,)})), id="radarr"),
        ],
    )
    def test_a_preferred_public_release_is_grabbed_in_place_of_an_outdated_private_copy(
        self, library: _Library
    ) -> None:
        # Pub's release is preferred, not a fallback, so the title is cached with no fallback marker.
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Pub", "S01E01", size=555))

        outcome = _outcome(entry, library, private_releases="fallback")

        assert outcome == _Outcome(
            added=("Pub",),
            details=(_Detail("skipped", "Priv private-only; grabbing public release Pub instead"),),
            cached=True,
        )

    def test_of_two_private_releases_only_the_one_nothing_replaces_warns(self) -> None:
        # The arr has PrivA's S01E01 at a size SeaDex no longer lists, and PubA's preferred release is grabbed in its
        # place. S02E01 has no file, and only PrivB's private release has it.
        entry = _entry(
            _private("PrivA", "S01E01", size=999),
            _private("PrivB", "S02E01", size=777),
            _public("PubA", "S01E01", size=555),
        )
        library = _sonarr(sonarr_ep(1, 1, size=100, release_group="PrivA"), _missing(2, 1))

        outcome = _outcome(entry, library, private_releases="fallback")

        assert outcome == _Outcome(
            added=("PubA",),
            details=(
                _Detail("skipped", "PrivA private-only; grabbing public release PubA instead"),
                _Detail("skipped", f"PrivB {PLAIN_NOTICE}", Severity.WARNING),
            ),
            needs_action=(_Row("PrivB", NO_FALLBACK_ROW, NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK),),
        )

    def test_a_group_whose_public_release_the_arr_has_warns_of_its_outdated_private_one(self) -> None:
        # The arr has M's public S01E01, and M's S02E01 at a size SeaDex doesn't list. F's fallback covers both
        # episodes but never replaces M's copy, so M's private release is skipped when added.
        entry = _entry(
            _public("M", "S01E01", size=555),
            _private("M", "S01E01", "S02E01", size=999),
            _public("F", "S01E01", "S02E01", size=777, preferred=False),
        )
        library = _sonarr(sonarr_ep(1, 1, size=555, release_group="M"), sonarr_ep(2, 1, size=100, release_group="M"))

        outcome = _outcome(entry, library, private_releases="fallback")

        assert outcome == _Outcome(
            skips=(_Skip("M", SkipReason.PRIVATE_ONLY),),
            needs_action=(_Row("M", OUTDATED_ROW, NeedsActionCause.PRIVATE_ONLY_OUTDATED),),
        )


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
