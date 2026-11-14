"""What SeaDex lists and what Pearlarr plans to grab: release groups, releases, skip reasons, and episode keys."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, NamedTuple

from seadex import Tag, TorrentRecord, Tracker

from .tvdb_orders import order_display_name


class EpisodeKey(NamedTuple):
    """The converted `(season, episode)` index key, always concrete ints."""

    season: int
    episode: int

    @property
    def label(self) -> str:
        """The key as `S01E02`."""

        return f"S{self.season:02d}E{self.episode:02d}"


@dataclass(frozen=True, slots=True)
class ReleaseEpisode:
    """One file of a release on one episode the placement put it on: the episode's numbers and the file's size.

    A file on two episodes is two of these, and so are two files on one episode.
    """

    season: int
    episode: int
    size: int

    @property
    def key(self) -> EpisodeKey:
        """The episode's `(season, episode)`."""

        return EpisodeKey(self.season, self.episode)


def normalized_infohash(raw: str | None) -> str | None:
    """The one spelling of an infohash: stripped and lowercase, a blank one None.

    An empty `hashes` filter matches every torrent in the qBittorrent dedup, and the pending import store keys on the
    hash byte-exact.
    """

    if raw is None:
        return None
    return raw.strip().lower() or None


def ignore_tag_set(tags: Iterable[str]) -> frozenset[str]:
    """The configured ignore tags, casefolded the way `carries_ignored_tag` compares them."""

    return frozenset(tag.casefold() for tag in tags)


def carries_ignored_tag(torrent: TorrentRecord, ignore_tags: frozenset[str]) -> bool:
    """Whether the torrent has any of the `ignore_tags`, compared casefolded on both sides."""

    return not ignore_tags.isdisjoint(tag.casefold() for tag in torrent.tags)


class SkipReason(StrEnum):
    """Why a release is skipped this run: at add time for its tracker, or by the placement.

    A member's name, lowercased, is the JSON `reason` of `release_skipped`. Its value is the phrase a message uses.
    """

    PRIVATE_ONLY = auto()
    UNSUPPORTED_TRACKER = auto()
    TRACKER_NOT_SELECTED = auto()
    MISNUMBERED = "misnumbered"
    """A torrent with a `PlacementVerdict.MISNUMBERED` file of the title's window: a file whose numbers disagree,
    or a specials torrent numbered another way than its listing. Grabbed, it would import onto the wrong episodes, so
    it's left for a hand import. Also a torrent with a file another TVDB order reads outside the title's window, where
    no listed title placed it (`PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW`), and nothing of it placed on the
    title."""
    EPISODE_TITLES_CONTRADICT = "episode titles contradict its numbers"
    """A torrent whose files' episode titles contradict their numbers, with nothing of it placed on the title
    (`PlacementVerdict.EPISODE_TITLES_CONTRADICT`): left for a hand import."""
    PLACED_ELSEWHERE = "files place under another title"
    """A release none of whose files land on the title, while some are another title's SeaDex lists the torrent
    with: placed or left for a hand import by that title's window, or stored on an episode of its window. That title's
    grab imports the placed ones, unless the title is in `ignore_anilist_ids`, which is never grabbed for. Not a
    hand-import skip: the release as a whole isn't the title's, and only its files nothing placed are a hand import."""
    PLACEMENT_INPUT_UNREAD = "a read failed"
    """A SeaDex listing for the series, Sonarr's parse of any of the release's video files while one is left to
    place, or the episode list of a series the pending import claims couldn't be read this run. Without the listing
    nothing is placed, since another title it names may have some of the files. Grabbing now could place a file on the
    wrong episode for good, or leave one unplaced that the import never places, so the release waits for the next
    run."""

    @property
    def left_for_hand_import(self) -> bool:
        """Whether the torrent is left for a hand import."""

        return self in _HAND_IMPORT_SKIPS


_HAND_IMPORT_SKIPS = frozenset({SkipReason.MISNUMBERED, SkipReason.EPISODE_TITLES_CONTRADICT})


@dataclass(frozen=True, slots=True, kw_only=True)
class AnotherTitlesCounts:
    """How the files of a release skipped as another title's (`SkipReason.PLACED_ELSEWHERE`) split, for its skip line.

    Built by keyword, so the two counts can't silently transpose.
    """

    another_titles: int
    """The files that are another title's, one SeaDex lists the torrent with: placed or left for a hand import by
    its window, or stored on its episodes."""

    unplaced: int
    """The other files, which nothing placed."""


NO_ANOTHER_TITLES_COUNTS = AnotherTitlesCounts(another_titles=0, unplaced=0)
"""The counts of a release not skipped as another title's."""


@dataclass
class Release:
    """One SeaDex release within a release group."""

    url: str = ""
    files: list[str] = field(default_factory=list[str])
    file_sizes: list[int] = field(default_factory=list[int])
    tracker: Tracker = Tracker.OTHER
    """A SeaDex `Tracker` object (not a str). The notifier renders it as the link text of a grab embed."""
    is_public: bool = True
    is_dual_audio: bool = False
    infohash: str | None = None
    """SeaDex's hash for the release, the key of its listing."""
    cross_seed_hash: str | None = None
    """The hash of a torrent already in qBittorrent with exactly this release's files under another hash.
    `TorrentService.mark_cross_seeds` sets it before anything reads the release."""
    to_grab: bool = False
    is_fallback: bool = False
    """A fallback, grabbed because the preferred release is private-only."""
    is_upgrade: bool = False
    """A grab over files the Arr has at an outdated size or on too many episodes. Never set without `to_grab`."""
    episodes: list[ReleaseEpisode] = field(default_factory=list[ReleaseEpisode])
    skip_reason: SkipReason | None = None
    """Why the placement skips the release this run (`attach_placements` sets it), None when it may be grabbed."""
    another_titles_counts: AnotherTitlesCounts = NO_ANOTHER_TITLES_COUNTS
    """How the release's files split when it's skipped as another title's (`PlacedRelease.another_titles_counts`)."""

    def __post_init__(self) -> None:
        self.infohash = normalized_infohash(self.infohash)

    @property
    def qbit_hash(self) -> str | None:
        """The hash the release's torrent goes by in qBittorrent: the cross-seed's when one has its files, else SeaDex's."""

        # Invariant: a release's pending import is keyed by the torrent that has its files in qBittorrent, which is
        # SeaDex's torrent unless a cross-seed was already there. The wait and the import read it by that key.
        return self.cross_seed_hash or self.infohash

    @property
    def left_for_hand_import(self) -> bool:
        """Whether the placement left the release's torrent for a hand import (`SkipReason.left_for_hand_import`)."""

        return self.skip_reason is not None and self.skip_reason.left_for_hand_import

    def plan_grab(self, *, upgrade: bool = False) -> None:
        """Mark the release to grab, `upgrade` marking a replacement of the arr's files. Never unsets an upgrade."""

        self.to_grab = True
        if upgrade:
            self.is_upgrade = True

    def drop_grab(self) -> None:
        """Clear the grab. `is_upgrade` clears with `to_grab`: it describes the grab, so it must never outlive it."""

        self.to_grab = False
        self.is_upgrade = False


@dataclass
class GroupReleases:
    """One SeaDex release group's releases, keyed by url under `releases`."""

    releases: dict[str, Release] = field(default_factory=dict[str, Release])
    tags: frozenset[Tag] = field(default_factory=frozenset[Tag])
    all_episodes: list[ReleaseEpisode] | None = None
    """`None` until `attach_placements` has run, distinct from an empty list (it ran and found nothing)."""


SeadexDict = dict[str, GroupReleases]
"""SeaDex release groups keyed by group name."""


class PlannedGrab(NamedTuple):
    """One release marked to grab that carries an infohash."""

    group: str
    """The release group's name, the key of `SeadexDict`."""
    url: str
    """The key under `GroupReleases.releases`."""
    item: Release
    """The release, the `to_grab` and file list the planner read."""
    qbit_hash: str
    """The hash its torrent goes by in qBittorrent (`Release.qbit_hash`), lowercase."""


def releases_to_grab(seadex_dict: SeadexDict) -> list[PlannedGrab]:
    """The releases marked to grab that carry an infohash. Narrow at the callers, never here."""

    return [
        PlannedGrab(srg, url, release, release.qbit_hash)
        for srg, group_releases in seadex_dict.items()
        for url, release in group_releases.releases.items()
        if release.to_grab and release.qbit_hash
    ]


# Included in the config's selection hash: bump when the release-selection
# rules change in code, or a fix leaves earlier decisions in doubt, so every
# cached title re-checks once after an upgrade.
SELECTION_RULES_VERSION: int = 17

SONARR_MISSING_KEY: int = 999
"""Out-of-range stand-in for a missing Sonarr `seasonNumber`/`episodeNumber`, never colliding with a real one."""


@dataclass(frozen=True, slots=True)
class OrderOnlyKey:
    """An episode that another TVDB order of a series numbers and Sonarr doesn't list: one half of a split episode, say."""

    order: str
    """The order's type name, as TVDB and SkyHook spell it (`dvd`, `alternate`, ...)."""

    key: EpisodeKey
    """The `(season, episode)` the order gives it."""

    @property
    def label(self) -> str:
        """The key under its order's name, as `DVD S07E01`."""

        return f"{order_display_name(self.order)} {self.key.label}"

    def to_json(self) -> dict[str, Any]:
        """The plain dict persisted inside a pending import."""

        return {"order": self.order, "season": self.key.season, "episode": self.key.episode}

    @classmethod
    def from_json(cls, raw: Any) -> "OrderOnlyKey | None":
        """Rebuild from the persisted dict, or None when `raw` isn't one."""

        match raw:
            case {"order": str(order), "season": int(season), "episode": int(episode)}:
                return cls(order, EpisodeKey(season, episode))
            case _:
                return None


def season_episode_key(season: int | None, episode: int | None) -> EpisodeKey:
    """The `EpisodeKey` for a possibly-missing pair, collapsing to the sentinel."""

    return EpisodeKey(
        season if season is not None else SONARR_MISSING_KEY,
        episode if episode is not None else SONARR_MISSING_KEY,
    )


def normalize_group(group: str | None) -> str | None:
    """Normalize a release group for comparison: strip whitespace/wrapping dashes, casefold.

    None for a missing or empty name.
    """

    if not group:
        return None
    return group.strip().strip("-").casefold()
