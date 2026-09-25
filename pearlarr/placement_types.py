"""Types the placement modules share: the episode index, verdicts, one file's placement, the batch, and the scope."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple, Self

from .manual_import import EntryNames
from .seadex_types import EpisodeKey, ParsedFileInfo, SonarrEpisode, index_episodes_by_key


@dataclass(frozen=True, slots=True)
class EpisodeIndex:
    """The import family's one episode index, built once per episode fetch.

    Both mappings are copied and made read-only at construction. `episode_index` is
    the one builder. Episodes with a falsy id (0) are dropped from EVERY facet
    before keying: a 0 id can never be POSTed to Sonarr, and a real-id twin
    behind one must still win its `(season, episode)` key. The planner's
    `index_episodes_by_key` deliberately KEEPS them (a 0-id file is still
    identity evidence). Do not unify the two.
    """

    by_id: Mapping[int, SonarrEpisode]
    """Episode id -> episode, in the fetch's (season) order. `list(by_id)` is
    the scope id list the grab stores on each entry's claim."""

    id_by_key: Mapping[EpisodeKey, int]
    """`(season, episode)` -> episode id (missing numbers collapse to
    `SONARR_MISSING_KEY`, first record wins, via `index_episodes_by_key`)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dicts, then wrap read-only.
        object.__setattr__(self, "by_id", MappingProxyType(dict(self.by_id)))
        object.__setattr__(self, "id_by_key", MappingProxyType(dict(self.id_by_key)))


def episode_index(ep_list: Iterable[SonarrEpisode]) -> EpisodeIndex:
    """Fold one `/api/v3/episode` fetch into the `EpisodeIndex` facets."""

    with_ids = [ep for ep in ep_list if ep.id]
    return EpisodeIndex(
        by_id={ep.id: ep for ep in with_ids},
        id_by_key={key: ep.id for key, ep in index_episodes_by_key(with_ids).items()},
    )


class PlacementVerdict(StrEnum):
    """How one file left `assign_episode_ids`: placed by which pass, excluded, or unplaced."""

    EXACT = "exact"
    """Its own `(season, episode)`, or Sonarr's matched pair, resolved inside the scope."""
    RELEASE_RUN = "release run"
    """A numbered run's own numbers placed it onto the run window, overriding Sonarr's inconsistent reading."""
    ABSOLUTE = "absolute"
    """Paired with an open id in absolute-number order (the absolute zip)."""
    SINGLE = "single file"
    """The only open file, with no episode number, onto the only open id."""
    ORDERED = "ordered"
    """A batch of numberless files no earlier pass touched, paired with the open ids in natural name order."""
    NUMBERED_RUN = "numbered run"
    """A `1..N` run among the files Sonarr found no episode for, paired with the run window."""
    TITLED = "titled"
    """The numberless file an entry's AniList title names, onto the only open id."""
    EPISODE_TITLE = "episode title"
    """Onto the one episode whose title its name carries, whatever its number said."""
    FOREIGN = "other slice"
    """Reads wholly outside the scope, or is titled as an episode outside it and confirmed by its reading or by no
    open id being left: never this record's to import."""
    DUPLICATE = "duplicate"
    """Reads inside the scope onto episodes other files provably hold, or is titled as an episode a version of the
    same file holds."""
    EXTRA = "extra"
    """An opening, an ending, a preview, a menu, a commercial, a trailer, a teaser, or a promo: never an episode."""
    HELD = "held"
    """A file of the picked release run while a parse is unknown: no other pass may place it, and it is re-checked."""
    SKIPPED = "skipped"
    """Nothing placed it and nothing proved it foreign."""

    @property
    def placed(self) -> bool:
        """Whether the verdict carries episode ids."""

        return self in _PLACED

    @property
    def excluded(self) -> bool:
        """Whether the file is knowably never this record's to import."""

        return self in _EXCLUDED


_PLACED = frozenset(
    {
        PlacementVerdict.EXACT,
        PlacementVerdict.RELEASE_RUN,
        PlacementVerdict.ABSOLUTE,
        PlacementVerdict.SINGLE,
        PlacementVerdict.ORDERED,
        PlacementVerdict.NUMBERED_RUN,
        PlacementVerdict.TITLED,
        PlacementVerdict.EPISODE_TITLE,
    }
)
_EXCLUDED = frozenset({PlacementVerdict.FOREIGN, PlacementVerdict.DUPLICATE, PlacementVerdict.EXTRA})


class Placement(NamedTuple):
    """One file's verdict, with its episode ids when placed."""

    name: str
    """The normalized basename."""
    ids: tuple[int, ...]
    """The episode ids, in claim order (empty unless `verdict.placed`)."""
    verdict: PlacementVerdict
    """How the file left the passes."""


class EpisodeAssignment(NamedTuple):
    """The outcome of assigning a torrent's on-disk files to resolved episode ids, one verdict per file."""

    placements: tuple[Placement, ...]
    """One per distinct name in `PlacementBatch.to_place`, batch order."""

    @property
    def assigned(self) -> dict[str, list[int]]:
        """Normalized basename -> episode ids for every placed file (each id in the scope, used once)."""

        return {p.name: list(p.ids) for p in self.placements if p.verdict.placed}

    @property
    def skipped(self) -> tuple[str, ...]:
        """The files nothing placed and nothing excluded: the caller warns, never guesses."""

        return tuple(p.name for p in self.placements if not p.verdict.placed and not p.verdict.excluded)

    @property
    def excluded(self) -> tuple[Placement, ...]:
        """The files this record knowably never imports (another slice's, a duplicate, an extra), and how."""

        return tuple(p for p in self.placements if p.verdict.excluded)

    @property
    def unplaced(self) -> tuple[str, ...]:
        """Every file without ids: the skips plus the exclusions."""

        return tuple(p.name for p in self.placements if not p.verdict.placed)


class PlacementBatch(NamedTuple):
    """A torrent's files still to place, with the parses of the WHOLE torrent.

    `parsed` may cover MORE files than `to_place`: seeded and gone files count as evidence (readings,
    titles, the shared-absolute check) but are never placed.
    """

    to_place: Sequence[str]
    """Normalized basenames in SeaDex order (the order only fixes deterministic output)."""

    parsed: Mapping[str, ParsedFileInfo | None]
    """Series-agnostic parse per file (None when Sonarr's parse was unavailable
    and no SxxExx fell out of the name)."""

    @property
    def names(self) -> tuple[str, ...]:
        """The distinct names to place, batch order."""

        return tuple(dict.fromkeys(self.to_place))

    @property
    def torrent_names(self) -> tuple[str, ...]:
        """Every distinct name of the torrent: the names to place, then those only `parsed` covers (seeded, gone)."""

        return tuple(dict.fromkeys([*self.to_place, *self.parsed]))

    @property
    def all_parses_known(self) -> bool:
        """Every parse came from Sonarr this run: no transport miss (None) and no offline `SxxExx` stand-in.

        Seeded and gone files are parsed by name too, so a miss on any of them counts, as
        does a file to place that has no parse entry.
        """

        return all((info := self.parsed.get(name)) is not None and not info.offline for name in self.torrent_names)


@dataclass(frozen=True, slots=True)
class TargetScope:
    """The episode set a placement batch may assign into.

    `resolved` is the FULL id list and `used` the ids already held, by seeded files or an earlier window.
    They stay separate so a fully seeded record still enforces its scope, while an
    EMPTY `resolved` means no scope at all (`unscoped`): the exact pass then places
    a name's own keys against the whole series map instead of placing nothing forever.
    """

    resolved: Sequence[int]
    """The entry's resolved episode ids, season order, seeded included, held as a tuple. A stray
    zero id still makes the scope real. It is never placed."""

    series: EpisodeIndex
    """ALL the series' episodes: the `(season, episode)` map every key resolves
    through (membership in `resolved` does the scoping, so an exact parse
    outside our entry is rejected), plus the titles and absolute numbers the
    run evidence reads."""

    used: frozenset[int] = frozenset()
    """Ids already held, by seeded files or by an earlier window's placements. Never placed again."""

    names: EntryNames = field(default_factory=EntryNames)
    """The series and AniList titles. Episode titles and extras words are read against them, and they break
    ties when several runs or numberless files could fill the open ids."""

    real_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """The resolved ids that can be placed. A stray zero keeps the scope real but is never one."""

    def __post_init__(self) -> None:
        # Detached from the caller's list: `real_ids` is derived from it once.
        object.__setattr__(self, "resolved", tuple(self.resolved))
        object.__setattr__(self, "real_ids", frozenset(i for i in self.resolved if i))

    @property
    def id_by_key(self) -> Mapping[EpisodeKey, int]:
        """`(season, episode)` -> id over the whole series."""

        return self.series.id_by_key

    @property
    def unscoped(self) -> bool:
        """No resolved set to scope against at all (never just fully seeded)."""

        return not self.resolved

    def admits(self, ep_id: int) -> bool:
        """Whether a file may be placed on the episode: one of the resolved set, or any when unscoped."""

        return self.unscoped or ep_id in self.real_ids

    def using(self, ids: Iterable[int]) -> Self:
        """The scope with `ids` used too: the ones it admits (every one when unscoped)."""

        return replace(self, used=self.used | frozenset(i for i in ids if self.admits(i)))
