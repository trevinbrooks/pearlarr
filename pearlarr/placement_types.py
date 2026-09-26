"""Types the placement modules share: the episode index, verdicts, one file's placement, the batch, and the scope."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple, Self

from .manual_import import NO_IDENTIFIED, EntryNames, IdentifiedNames
from .seadex_types import NO_ALIASES, EpisodeKey, ParsedFileInfo, SonarrEpisode, SpecialAliases, index_episodes_by_key


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


def all_specials(series: EpisodeIndex, ids: Iterable[int]) -> bool:
    """Whether every id is one of the series' specials (season 0)."""

    return all((ep := series.by_id.get(ep_id)) is not None and ep.season_number == 0 for ep_id in ids)


@dataclass(frozen=True, slots=True)
class SeriesFacts:
    """Lookups built once from a series' episode index, for placement to read. Nothing changes them after."""

    series: EpisodeIndex

    key_by_id: Mapping[int, EpisodeKey] = field(init=False, repr=False, compare=False)
    """Episode id -> `(season, episode)`."""
    season_counts: Mapping[int, int] = field(init=False, repr=False, compare=False)
    """Season -> how many episodes it has."""
    absolute_of: Mapping[int, int] = field(init=False, repr=False, compare=False)
    """Episode id -> absolute number, for the episodes that have one."""

    def __post_init__(self) -> None:
        keys = self.series.id_by_key
        object.__setattr__(self, "key_by_id", MappingProxyType({ep_id: key for key, ep_id in keys.items()}))
        object.__setattr__(self, "season_counts", MappingProxyType(Counter(key.season for key in keys)))
        absolute_of = {
            ep_id: ep.absolute_episode_number
            for ep_id, ep in self.series.by_id.items()
            if ep.absolute_episode_number is not None
        }
        object.__setattr__(self, "absolute_of", MappingProxyType(absolute_of))

    @property
    def id_by_key(self) -> Mapping[EpisodeKey, int]:
        """`(season, episode)` -> episode id, straight from the index."""

        return self.series.id_by_key


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
    IDENTIFIED = "listed size"
    """Placed by file size: SeaDex lists a single-file release of this size for the episode, and the name agrees or
    has no number."""
    OVERRIDDEN = "listed size over its name"
    """Placed by file size even though the name's number points at a different episode. The caller logs a warning."""
    ALTERNATE = "alternate numbering"
    """A file of a misnumbered specials pack, placed on the special its number's alias points at."""
    ALIASED_ELSEWHERE = "aliased special outside the window"
    """Not placed here: a file of a misnumbered specials pack whose alias points at a special outside this window.
    A claim, not a refusal, so the window that holds that special may still place it."""
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
    MISNUMBERED = "misnumbered"
    """A file of a specials pack whose numbers don't match the windows listing it, when its special aliases can't
    place it either or another window places it elsewhere. Never placed by its numbers, left for a hand import."""
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

    @property
    def refused(self) -> bool:
        """Whether the verdict settles the file with no ids: held for a re-ask, or misnumbered for a hand import."""

        return self in _REFUSED


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
        PlacementVerdict.IDENTIFIED,
        PlacementVerdict.OVERRIDDEN,
        PlacementVerdict.ALTERNATE,
    }
)
_EXCLUDED = frozenset({PlacementVerdict.FOREIGN, PlacementVerdict.DUPLICATE, PlacementVerdict.EXTRA})
_REFUSED = frozenset({PlacementVerdict.HELD, PlacementVerdict.MISNUMBERED})


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


type SizeIdentities = Mapping[int, int]
"""File size -> episode id, from SeaDex releases with a single video file listed under an entry that covers one
episode. A file with that exact size is that episode."""


@dataclass(frozen=True, slots=True)
class TorrentListing:
    """The episodes covered by the SeaDex entries that list one torrent, and those entries' special aliases."""

    ids: frozenset[int]
    """Union of the episode windows of every entry in the series that lists the torrent."""

    special_aliases: SpecialAliases = NO_ALIASES
    """The union of those entries' special aliases, minus any TMDB number two entries pair differently. Empty when
    the entries pair against different TMDB shows."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "special_aliases", MappingProxyType(dict(self.special_aliases)))


EMPTY_LISTING = TorrentListing(frozenset())
"""A torrent no entry lists. Listings match by infohash, so a url without one always gets this."""


@dataclass(frozen=True, slots=True)
class ListingEvidence:
    """What SeaDex's listings for the series tell the placer about one torrent.

    Import time only fills in `identified`.
    """

    listing: TorrentListing = EMPTY_LISTING
    """The torrent's listing: the ids a numbered specials pack is checked against, and the special aliases that
    can place a misnumbered one. Empty at import time, which turns the pack check off."""

    identified: IdentifiedNames = NO_IDENTIFIED
    """File name -> episode id, for the files whose size matched an episode (see `SizeIdentities`)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "identified", MappingProxyType(dict(self.identified)))

    @property
    def ids(self) -> frozenset[int]:
        """Every episode id covered by the entries that list this torrent, this entry included."""

        return self.listing.ids

    @property
    def special_aliases(self) -> SpecialAliases:
        """The listing's TMDB -> TVDB specials numbers (`TorrentListing.special_aliases`)."""

        return self.listing.special_aliases


NO_EVIDENCE = ListingEvidence()
"""Nothing from the listings: no ids and no size matches, so the size placement and the specials pack check do
nothing."""

type SeededNames = Mapping[str, tuple[int, ...]]
"""Normalized file name -> the episode ids the stored record already holds it on."""

NO_SEEDED: SeededNames = MappingProxyType({})
"""No seeded files: a torrent with no stored record, or a scope `place_leftover` hasn't filled in yet."""


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

    listing: ListingEvidence = NO_EVIDENCE
    """What SeaDex's listings tell us about this torrent. At import time only `identified` is filled in."""

    seeded: SeededNames = NO_SEEDED
    """The stored record's map, name -> ids, the same on every window (`place_leftover` fills it in). Seeded files
    are never placed again, but the misnumbered pack pass checks each seeded member sits on its aliased special."""

    real_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """The resolved ids that can be placed. A stray zero keeps the scope real but is never one."""

    def __post_init__(self) -> None:
        # Detached from the caller's list and dict: `real_ids` is derived from the list once.
        object.__setattr__(self, "resolved", tuple(self.resolved))
        object.__setattr__(self, "real_ids", frozenset(i for i in self.resolved if i))
        object.__setattr__(self, "seeded", MappingProxyType(dict(self.seeded)))

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

    @property
    def specials_window(self) -> bool:
        """A scoped window of specials only, the one shape a listing judges."""

        return not self.unscoped and all_specials(self.series, self.real_ids)

    def specials_listing(self) -> frozenset[int] | None:
        """The listing a numbered pack is judged by: read, all specials, covering the window. None stands it down."""

        listed = self.listing.ids
        if not (self.specials_window and listed and self.real_ids <= listed):
            return None
        return listed if all_specials(self.series, listed) else None


@dataclass(frozen=True, slots=True)
class ListingsRead:
    """What this run read from one series' SeaDex entries: listings by infohash, plus the size identities."""

    by_hash: Mapping[str, TorrentListing | None]
    """Infohash -> listing, or None if an entry listing that torrent couldn't be read."""

    identities: SizeIdentities | None
    """The series' size identities, minus any size that two entries give to different episodes. None if any
    entry's episodes couldn't be read, since that entry's sizes would be missing."""

    def __post_init__(self) -> None:
        # Detach from the caller's dicts, then wrap read-only.
        object.__setattr__(self, "by_hash", MappingProxyType(dict(self.by_hash)))
        if self.identities is not None:
            object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))

    def listing(self, infohash: str) -> TorrentListing | None:
        """The torrent's listing: None if it couldn't be read, `EMPTY_LISTING` if no entry lists it."""

        return self.by_hash.get(infohash, EMPTY_LISTING)
