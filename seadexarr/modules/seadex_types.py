"""The ``seadex_dict`` domain vocabulary: the shapes the planner and notifier read.

The central ``seadex_dict`` is a four-level mapping built once per AniList entry
in :meth:`seadexarr.modules.seadex_arr.SeaDexArr.get_seadex_dict` (around
seadex_arr.py:368-376) and threaded through the decision engine
(:mod:`seadexarr.modules.planner`) and the Discord notifier
(:mod:`seadexarr.modules.notify`). The two keyed levels stay plain ``dict``\\ s
(release groups keyed by name, urls keyed by url string), but the *value
records* at each level are modeled as :func:`dataclasses.dataclass`: a real
domain model with attribute access (``item.download`` rather than
``item["download"]``) and defaults that make a partially built record legal. Each
field carries a default because the records are filled in across construction
stages (``episodes``/``all_episodes`` are appended later by the episode parser,
``download`` is flipped per call), so a freshly built record need not pass every
field.

The defaults also encode one load-bearing distinction:
``SeadexReleaseGroupItem.all_episodes`` is ``None`` when no episode parsing ran
(Radarr movies) and an empty ``list`` when parsing ran but found nothing -
``get_same_files_groups`` keys off exactly that difference.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, Self, runtime_checkable

from seadex import Tag, Tracker


@dataclass
class EpisodeRecord:
    """One parsed ``{season, episode, size}`` record for a SeaDex file.

    ``season``/``episode`` default to ``None`` (a record built without them - only
    seen in characterization tests - reduces to a never-matching ``(None, None)``
    key, which can never collide with a real Arr episode).
    """

    season: int | None = None
    episode: int | None = None
    size: int = 0


@dataclass
class SeadexUrlItem:
    """One SeaDex url record within a release group.

    ``tracker`` holds a SeaDex ``Tracker`` object (not a str); it defaults to
    ``None`` because the test builders don't supply one and no in-scope consumer
    reads it.
    """

    url: str = ""
    files: list[str] = field(default_factory=list)
    size: list[int] = field(default_factory=list)
    tracker: Tracker = Tracker.OTHER
    is_public: bool = True
    hash: str | None = None
    download: bool = False
    episodes: list[EpisodeRecord] = field(default_factory=list)


@dataclass
class SeadexReleaseGroupItem:
    """One SeaDex release-group record, keyed by url under ``urls``.

    ``all_episodes`` is ``None`` until the episode parser has run:
    ``get_same_files_groups`` deliberately distinguishes ``None`` (no episode
    parsing, e.g. Radarr) from an empty list (parsing ran but found nothing).
    """

    urls: dict[str, SeadexUrlItem] = field(default_factory=dict)
    tags: frozenset[Tag] = field(default_factory=frozenset)
    all_episodes: list[EpisodeRecord] | None = None


SeadexDict = dict[str, SeadexReleaseGroupItem]
"""The central object: SeaDex release groups keyed by group name."""


class EmbedField(NamedTuple):
    """A Discord embed field, typed inside the notifier.

    The ``discordwebhook`` library wants plain ``{"name", "value"}`` dicts at the
    JSON boundary, so :meth:`to_dict` serializes the field back to that exact
    shape when :meth:`seadexarr.modules.notify.Notifier.build_fields` returns.
    """

    name: str
    value: str

    def to_dict(self) -> dict[str, str]:
        """The plain dict the Discord webhook payload expects."""

        return {"name": self.name, "value": self.value}


SONARR_MISSING_KEY: int = 999
"""Out-of-range fallback for a missing Sonarr ``seasonNumber``/``episodeNumber``.

Used when indexing Sonarr episodes by (season, episode); it never collides with
a real key, so an episode with a missing key simply fails to match.
"""


def as_size_list(size: int | list[int | None] | None) -> list[int]:
    """Normalize a size value to a list of concrete sizes.

    ``None`` (or a missing size) becomes ``[]``; a bare int becomes ``[int]``; a
    list is copied with any ``None`` entries dropped (a ``None`` size carries no
    size to compare). The single home for the size-as-list coercion the planner
    used to inline.
    """

    if size is None:
        return []
    if isinstance(size, int):
        return [size]
    return [s for s in size if s is not None]


# --- Arr items (Sonarr series / Radarr movies) ------------------------------

@runtime_checkable
class ArrItem(Protocol):
    """The attribute surface shared by a Sonarr series and a Radarr movie."""

    id: int
    title: str
    imdbId: str | None
    monitored: bool


@runtime_checkable
class SonarrItem(ArrItem, Protocol):
    """A Sonarr series item: an :class:`ArrItem` keyed on ``tvdbId``."""

    tvdbId: int


@runtime_checkable
class RadarrItem(ArrItem, Protocol):
    """A Radarr movie item: an :class:`ArrItem` keyed on ``tmdbId``."""

    tmdbId: int


# --- Sonarr episodes (``/api/v3/episode`` JSON) -----------------------------

@dataclass(frozen=True, slots=True)
class SonarrEpisodeFile:
    """The ``episodeFile`` sub-record of a Sonarr episode."""

    release_group: str | None = None
    size: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw Sonarr ``episodeFile`` dict."""

        return cls(release_group=raw.get("releaseGroup"), size=raw.get("size"))


@dataclass(frozen=True, slots=True)
class SonarrEpisode:
    """One Sonarr ``/api/v3/episode`` record, parsed at the client boundary."""

    season_number: int | None = None
    episode_number: int | None = None
    episode_file_id: int = 0
    monitored: bool = True
    episode_file: SonarrEpisodeFile | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw Sonarr episode dict (filters unknown keys)."""

        raw_file = raw.get("episodeFile")
        return cls(
            season_number=raw.get("seasonNumber"),
            episode_number=raw.get("episodeNumber"),
            episode_file_id=raw.get("episodeFileId", 0),
            monitored=raw.get("monitored", True),
            episode_file=SonarrEpisodeFile.from_api(raw_file) if raw_file else None,
        )


type ArrReleaseDict = dict[str | None, list[int | None]]
"""Release group (``None`` when unknown) -> its existing-file sizes.

Built by the strategies (Sonarr accumulates a per-episode size list; Radarr
wraps its single movie size in a one-element list) and read in the planner via
:func:`as_size_list`, which drops the ``None`` placeholders.
"""


type TvdbMappings = dict[int, list[tuple[int, int | None]]]
"""AniBridge TVDB season -> inclusive ``(start, end)`` episode ranges."""


# --- AniList Media node (cached GraphQL ``Media`` record) --------------------

@dataclass(frozen=True, slots=True)
class AniListMediaNode:
    """One AniList ``Media`` node, parsed once at the cache read boundary."""

    id: int | None = None
    title_english: str | None = None
    title_romaji: str | None = None
    episodes: int | None = None
    cover_image: str | None = None
    format: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from a raw AniList ``Media`` dict (``{}`` on a miss).

        Reads the nested ``title``/``coverImage`` sub-objects null-safely,
        mirroring the former chained ``.get(...) or {}`` walks: an English title
        is preferred with a romaji fallback handled by the caller, and the
        ``large`` cover variant is the one downstream reads.
        """

        title = raw.get("title") or {}
        cover = raw.get("coverImage") or {}
        return cls(
            id=raw.get("id"),
            title_english=title.get("english"),
            title_romaji=title.get("romaji"),
            episodes=raw.get("episodes"),
            cover_image=cover.get("large"),
            format=raw.get("format"),
        )
