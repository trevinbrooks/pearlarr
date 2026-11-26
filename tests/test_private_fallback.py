# pyright: strict
"""Tests for `seadex.private_releases`: what a title shows when SeaDex prefers a private release.

A private release is never grabbed. In warn mode, the default, one nothing public replaces warns, and leaves its title
uncached under a row saying why unless something else was grabbed for the title. In fallback mode the best public
releases SeaDex doesn't prefer are offered beside a private one whose files no preferred public release covers, and
grabbed in its place, but never in place of a copy the arr has at an outdated size. A private release nothing replaces
there always leaves its title uncached.

Each case checks a title the way a run does: the filter builds the entry's groups, the plan marks what to grab and
posts its private-only notices, and the grab adds, warns, files the needs-action row and caches or not. The choice
among groups covering the same files is `test_grab_planner.py`'s to prove, and the build `test_seadex_dict.py`'s.

The cases set each release's episodes from the episodes its file names number, where a Sonarr run places them with
the series, its orders and the name reader, none of which these cases are about.
"""

import re
from dataclasses import dataclass

import pytest
from seadex import EntryRecord, Tracker

from pearlarr.arr_models import ArrFiles, SonarrEpisode
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.output import NeedsActionCause, ReleaseSkipped, Severity, TitleDetail
from pearlarr.releases import PlacedEpisode, SeadexDict, SkipReason

from .builders import (
    AddStatus,
    FakeCacheStore,
    FakeTorrents,
    arr_files,
    grab_request,
    make_config,
    make_entry_record,
    make_grab_pipeline,
    make_planner,
    make_release_filter,
    make_run_deps,
    make_torrent_record,
    placed_episode,
    sonarr_ep,
    title_is_cached,
)
from .fakes import install_recording_hub

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


def _outcome(
    entry: EntryRecord, library: _Library, *, private_releases: str, cache: FakeCacheStore | None = None
) -> _Outcome:
    """The outcome of a run's check of `entry`'s title against `library` under `private_releases`.

    qBittorrent adds every public release asked for. The run stores the title's row in `cache`, for a later run to
    read.
    """

    cache = cache if cache is not None else FakeCacheStore()
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


class TestTheArrHasNoCopyOfTheTitle:
    """The arr has no file of the episodes SeaDex lists."""

    def test_warn_mode_warns_of_a_private_release_it_cannot_grab(self) -> None:
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))

        outcome = _outcome(entry, _sonarr(_missing(1, 1)), private_releases="warn")

        assert outcome == _Outcome(
            details=(_Detail("skipped", f"Priv {PLAIN_NOTICE}", Severity.WARNING),),
            needs_action=(_Row("Priv", PLAIN_ROW, NeedsActionCause.PRIVATE_ONLY),),
        )

    def test_fallback_mode_grabs_a_fallback_in_place_of_a_private_release(self) -> None:
        # Fall's release isn't preferred, so the title is cached as satisfied by a fallback.
        entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))

        outcome = _outcome(entry, _sonarr(_missing(1, 1)), private_releases="fallback")

        assert outcome == _Outcome(
            added=("Fall",),
            details=(_Detail("skipped", "Priv private-only; falling back to Fall"),),
            cached=True,
            fallback_marker=True,
        )

    @pytest.mark.parametrize(
        ("private_releases", "expected"),
        [
            pytest.param(
                "warn",
                _Outcome(added=("G",), skips=(_Skip("G", SkipReason.PRIVATE_ONLY),), cached=True),
                id="warn-caches-a-title-it-grabbed-for",
            ),
            pytest.param(
                "fallback",
                _Outcome(
                    added=("G",),
                    skips=(_Skip("G", SkipReason.PRIVATE_ONLY),),
                    needs_action=(_Row("G", NO_FALLBACK_ROW, NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK),),
                ),
                id="fallback-leaves-it-uncached",
            ),
        ],
    )
    def test_a_groups_private_release_is_skipped_beside_its_public_one(
        self, private_releases: str, expected: _Outcome
    ) -> None:
        # Only G's private release has S02E01, and no other group is listed to cover it, so G's public release is
        # added and its private one skipped when the grab comes to it.
        entry = _entry(_public("G", "S01E01", size=555), _private("G", "S01E01", "S02E01", size=999))

        outcome = _outcome(entry, _sonarr(_missing(1, 1), _missing(2, 1)), private_releases=private_releases)

        assert outcome == expected


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


def test_a_title_cached_by_a_fallback_grab_is_checked_again_by_a_warn_mode_run() -> None:
    """Switched to warn straight after a fallback's grab, a run checks the title again.

    The title's private-only warning can come back only if the run checks it.
    """

    entry = _entry(_private("Priv", "S01E01", size=999), _public("Fall", "S01E01", size=555, preferred=False))
    cache = FakeCacheStore()
    _outcome(entry, _sonarr(_missing(1, 1)), private_releases="fallback", cache=cache)
    warn_run = ArrRunServices(make_run_deps(config=make_config(private_releases="warn"), cache_store=cache), Arr.SONARR)

    unchanged = warn_run.is_unchanged(entry.anilist_id, entry, lambda: "")

    assert unchanged is False
