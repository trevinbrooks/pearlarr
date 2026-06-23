"""The ``seadex_dict`` domain vocabulary: the shapes the planner and notifier read.

The central ``seadex_dict`` is a four-level mapping built once per AniList entry
in :meth:`seadexarr.modules.seadex_arr.SeaDexArr.get_seadex_dict` (around
seadex_arr.py:368-376) and threaded through the decision engine
(:mod:`seadexarr.modules.planner`) and the Discord notifier
(:mod:`seadexarr.modules.notify`). The two keyed levels stay plain ``dict``\\ s
(release groups keyed by name, urls keyed by url string), but the *value
records* at each level are modelled as :func:`dataclasses.dataclass`: a real
domain model with attribute access (``item.download`` rather than
``item["download"]``) and defaults that make a partially-built record legal. Each
field carries a default because the records are filled in across construction
stages (``episodes``/``all_episodes`` are appended later by the episode parser,
``download`` is flipped per call), so a freshly built record need not pass every
field.

The defaults also encode one load-bearing distinction:
``SeadexReleaseGroupItem.all_episodes`` is ``None`` when no episode parsing ran
(e.g. Radarr movies) and an empty ``list`` when parsing ran but found nothing -
``get_same_files_groups`` keys off exactly that difference.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

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


def as_size_list(size: int | list[int] | None) -> list[int]:
    """Normalize a size value to a list of sizes.

    ``None`` (or a missing size) becomes ``[]``; a bare int becomes ``[int]``; a
    list passes through as a fresh list. The single home for the size-as-list
    coercion the planner used to inline.
    """

    if size is None:
        return []
    if isinstance(size, int):
        return [size]
    return list(size)
