"""Pure placement vocabulary: the episode index, verdicts, one file's placement, the assignment, batch, and scope."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple

from .manual_import import EntryNames
from .seadex_types import EpisodeKey, ParsedFileInfo, SonarrEpisode, index_episodes_by_key


@dataclass(frozen=True, slots=True)
class EpisodeIndex:
    """The import family's one episode index, built once per episode fetch.

    Both facets detach and wrap read-only at construction. `episode_index` is
    the one builder. Episodes with a falsy id (0) are dropped from EVERY facet
    before keying: a 0 id can never be POSTed to Sonarr, and a real-id twin
    behind one must still win its `(season, episode)` key. The planner's
    `index_episodes_by_key` deliberately KEEPS them (a 0-id file is still
    identity evidence). Do not unify the two.
    """

    by_id: Mapping[int, SonarrEpisode]
    """Episode id -> episode, in the fetch's (season) order. `list(by_id)` is
    the resolved set the add flow persists onto each seed."""

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
    """A `1..N` run's own numbering indexed the window over Sonarr's incoherent reading."""
    ABSOLUTE = "absolute"
    """The clean absolute zip."""
    SINGLE = "single file"
    """One numberless leftover onto one leftover episode."""
    ORDERED = "ordered"
    """The pristine numberless batch, zipped in natural name order."""
    NUMBERED_RUN = "numbered run"
    """A `1..N` run among the files Sonarr could not read at all."""
    TITLED = "titled"
    """The one numberless leftover an entry title names, onto one leftover episode."""
    FOREIGN = "other slice"
    """Resolves cleanly, entirely outside this record's set: never this record's to import."""
    DUPLICATE = "duplicate"
    """Resolves inside the set onto an episode another file already holds."""
    HELD = "held"
    """A release-run member whose batch has an unknown parse: no other pass may place it (re-asked)."""
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
    }
)
_EXCLUDED = frozenset({PlacementVerdict.FOREIGN, PlacementVerdict.DUPLICATE})


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
        """The files this record knowably never imports (another slice's, a refused duplicate), with their verdicts."""

        return tuple(p for p in self.placements if p.verdict.excluded)

    @property
    def unplaced(self) -> tuple[str, ...]:
        """Every file without ids: the skips plus the exclusions."""

        return tuple(p.name for p in self.placements if not p.verdict.placed)


class PlacementBatch(NamedTuple):
    """A torrent's leftover on-disk files to place, with the WHOLE batch's parses.

    `parsed` may cover MORE files than `to_place`: seeded and gone files feed
    the absolute leg's shared-absolute tell, but only `to_place` is ever placed.
    """

    to_place: Sequence[str]
    """Normalized basenames in SeaDex order (the order only fixes deterministic output)."""

    parsed: Mapping[str, ParsedFileInfo | None]
    """Series-agnostic parse per file (None when Sonarr's parse was unavailable
    and no SxxExx fell out of the name)."""

    @property
    def all_parses_known(self) -> bool:
        """Every parse came from Sonarr this run: no transport miss (None) and no offline `SxxExx` stand-in.

        Seeded and gone names ride the batch parsed by name, so a miss on any of them counts, as
        does a name to place the parses never covered.
        """

        names = dict.fromkeys([*self.to_place, *self.parsed])
        return all((info := self.parsed.get(name)) is not None and not info.offline for name in names)


class TargetScope(NamedTuple):
    """The episode set a placement batch may assign into.

    `resolved` is the FULL resolved set and `used` the ids seeds already own,
    kept separate so a fully seeded record stays scope-enforced while an EMPTY
    `resolved` means no scope at all (`unscoped`): the exact leg then places
    name-parsed pairs against the live series map instead of sticking forever.
    """

    resolved: Sequence[int]
    """The entry's resolved episode ids, season order, seeded included. A stray
    zero id still makes the scope real. It is never placed."""

    series: EpisodeIndex
    """ALL the series' episodes: the `(season, episode)` map every key resolves
    through (membership in `resolved` does the scoping, so an exact parse
    outside our entry is rejected), plus the titles and absolute numbers the
    run evidence reads."""

    used: frozenset[int] = frozenset()
    """Ids a seed already owns, never handed to a leftover file."""

    names: EntryNames = EntryNames()
    """The series and AniList titles: when several runs or numberless files could take the window, the one
    an AniList title names does."""

    @property
    def id_by_key(self) -> Mapping[EpisodeKey, int]:
        """`(season, episode)` -> id over the whole series."""

        return self.series.id_by_key

    @property
    def unscoped(self) -> bool:
        """No resolved set to scope against at all (never just fully seeded)."""

        return not self.resolved
