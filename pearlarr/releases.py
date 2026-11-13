"""What SeaDex lists and what Pearlarr plans to grab: release groups, urls, grab holds, and episode keys."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple

from seadex import Tag, TorrentRecord, Tracker

from .tvdb_orders import order_display_name


@dataclass
class ReleaseEpisode:
    """One parsed `{season, episode, size}` record for a SeaDex file."""

    season: int | None = None
    episode: int | None = None
    size: int = 0


def normalized_infohash(raw: str | None) -> str | None:
    """The one spelling of an infohash: stripped and lowercase, a blank one None.

    An empty `hashes` filter matches every torrent in the qBittorrent dedup, and the pending store keys on the hash
    byte-exact.
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


class PlacementSkip(StrEnum):
    """Why the placement refuses to grab a flagged url this run."""

    MISNUMBERED = "misnumbered"
    """A pack with a `PlacementVerdict.MISNUMBERED` file of the entry's own window: a file whose own numbers
    disagree, or a specials pack numbered another way than its listing. Grabbed, it would import onto the wrong
    episodes, so it's left for a hand import. Also a pack with a file another TVDB order reads outside the entry's
    window, where no listed title placed it (`PlacementVerdict.OTHER_ORDER_ELSEWHERE`), and nothing of it placed on
    the entry."""
    INPUT_UNREAD = "a read failed"
    """A SeaDex listing for the series, Sonarr's parse of any of the url's video files while one is left to place,
    or the episode list of a series the record claims couldn't be read this run. Without the listing nothing is
    placed, since another title it names may own some of the files. Grabbing now could seed a file onto the wrong
    episode for good, or leave one unplaced that the import never places, so the url waits for the next run."""
    TITLE_REFUTED = "titles refute its numbers"
    """A pack whose file titles contradict its numbers, with nothing of it placed on the entry
    (`PlacementVerdict.TITLE_REFUTED`): left for a hand import."""
    PLACED_ELSEWHERE = "files place under another title"
    """A url none of whose files land on the entry, while some are another title's SeaDex lists the torrent with:
    placed or left for a hand import by that title's window, or stored on an episode its window holds. That title's
    grab imports the placed ones, unless the title is in `ignore_anilist_ids`, which is never grabbed for. Not a
    hand-import hold: the url as a whole isn't the entry's, and only its files nothing placed are a hand import."""

    @property
    def left_for_hand_import(self) -> bool:
        """Whether the pack is left for a hand import."""

        return self in _HAND_IMPORT_SKIPS


_HAND_IMPORT_SKIPS = frozenset({PlacementSkip.MISNUMBERED, PlacementSkip.TITLE_REFUTED})


@dataclass(frozen=True, slots=True, kw_only=True)
class AnotherTitlesCounts:
    """How the files of a url held as another title's (`GrabHold.PLACED_ELSEWHERE`) split, for its skip line.

    Built by keyword, so the two counts can't silently transpose.
    """

    another_titles: int
    """The files another title SeaDex lists the torrent with holds: placed or left for a hand import by its window,
    or stored on its episodes."""

    unplaced: int
    """The other files, which nothing placed."""


NO_ANOTHER_TITLES_COUNTS = AnotherTitlesCounts(another_titles=0, unplaced=0)
"""The counts of a url not held as another title's."""


@dataclass
class Release:
    """One SeaDex url record within a release group."""

    url: str = ""
    files: list[str] = field(default_factory=list[str])
    file_sizes: list[int] = field(default_factory=list[int])
    tracker: Tracker = Tracker.OTHER
    """A SeaDex `Tracker` object (not a str). The notifier renders it as the link text of a grab embed."""
    is_public: bool = True
    is_dual_audio: bool = False
    infohash: str | None = None
    """SeaDex's hash for the url, the key of its listing."""
    cross_seed_hash: str | None = None
    """The hash of a torrent already in qBittorrent that holds exactly this url's files under another hash.
    `TorrentService.mark_cross_seeds` sets it before anything reads the item."""
    to_grab: bool = False
    is_fallback: bool = False
    """A public alternative added because the preferred release is private-only."""
    is_upgrade: bool = False
    """A grab over files the Arr holds at a stale size or on too many episodes. Never set without `download`."""
    episodes: list[ReleaseEpisode] = field(default_factory=list[ReleaseEpisode])
    skip_reason: PlacementSkip | None = None
    """Why the placement refuses to grab the url this run (`attach_placements` sets it), None when it may be."""
    another_titles_counts: AnotherTitlesCounts = NO_ANOTHER_TITLES_COUNTS
    """How the url's files split when it's held as another title's (`UrlPlacement.hold_counts`)."""

    def __post_init__(self) -> None:
        self.infohash = normalized_infohash(self.infohash)

    @property
    def qbit_hash(self) -> str | None:
        """The hash the url's torrent goes by in qBittorrent: the cross-seed's when one holds its files, else SeaDex's."""

        # Invariant: a url's record is keyed by the torrent that holds its files in qBittorrent, which is SeaDex's
        # torrent unless a cross-seed was already there. The wait and the import read the record by that key.
        return self.cross_seed_hash or self.infohash

    @property
    def left_for_hand_import(self) -> bool:
        """Whether the placement left the url's pack for a hand import (`GrabHold.hand_import`)."""

        return self.skip_reason is not None and self.skip_reason.left_for_hand_import

    def plan_grab(self, *, upgrade: bool = False) -> None:
        """Mark the url to grab, `upgrade` marking a replacement of held files. Never clears an upgrade already set."""

        self.to_grab = True
        if upgrade:
            self.is_upgrade = True

    def drop_grab(self) -> None:
        """Clear the grab. `upgrade` clears with `download`: it describes the grab, so it must never outlive it."""

        self.to_grab = False
        self.is_upgrade = False


@dataclass
class GroupReleases:
    """One SeaDex release-group record, keyed by url under `urls`."""

    releases: dict[str, Release] = field(default_factory=dict[str, Release])
    tags: frozenset[Tag] = field(default_factory=frozenset[Tag])
    all_episodes: list[ReleaseEpisode] | None = None
    """`None` until the episode parser has run, distinct from an empty list (it ran and found nothing)."""


SeadexDict = dict[str, GroupReleases]
"""SeaDex release groups keyed by group name."""


class PlannedGrab(NamedTuple):
    """One url flagged to grab that carries an infohash."""

    group: str
    """The release group's name, the key of `SeadexDict`."""
    url: str
    """The key under `SeadexReleaseGroupItem.urls`."""
    item: Release
    """The url's item, the flag and file list the planner read."""
    qbit_hash: str
    """The hash its torrent goes by in qBittorrent (`SeadexUrlItem.qbit_hash`), lowercase."""


def releases_to_grab(seadex_dict: SeadexDict) -> list[PlannedGrab]:
    """The urls flagged to grab that carry an infohash. Narrow at the riders, never here."""

    return [
        PlannedGrab(srg, url, release, release.qbit_hash)
        for srg, group_releases in seadex_dict.items()
        for url, release in group_releases.releases.items()
        if release.to_grab and release.qbit_hash
    ]


# Folded into the config's selection digest: bump when the release-selection
# rules change in code, or a fix leaves earlier verdicts in doubt, so every
# cached verdict re-checks once after an upgrade.
SELECTION_RULES_VERSION: int = 17

SONARR_MISSING_KEY: int = 999
"""Out-of-range stand-in for a missing Sonarr `seasonNumber`/`episodeNumber`, never colliding with a real one."""


class EpisodeKey(NamedTuple):
    """The folded `(season, episode)` index key, always concrete ints."""

    season: int
    episode: int

    @property
    def label(self) -> str:
        """The key as `S01E02`."""

        return f"S{self.season:02d}E{self.episode:02d}"


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
        """The plain dict persisted inside a pending record."""

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


def normalize_group(group: str) -> str:
    """Normalize a release group for comparison: strip whitespace/wrapping dashes, casefold."""

    return group.strip().strip("-").casefold()


def normalize_rg(name: str | None) -> str | None:
    """`normalize_group` with None-tolerance: None for a missing/blank name."""

    if not name:
        return None
    return normalize_group(name)
