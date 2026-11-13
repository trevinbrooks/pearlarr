"""Types the placement modules share: the episode index, verdicts, one file's placement, the batch, and the scopes."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Self

from .episode_titles import EpisodeTitles
from .manual_import import EntryNames, EntryWindow, FileEpisodeMap
from .seadex_types import (
    NO_ALIASES,
    EpisodeKey,
    OrderOnlyKey,
    ParsedFileInfo,
    SonarrEpisode,
    SpecialAliases,
    index_episodes_by_key,
)
from .tvdb_orders import NO_ORDERING, OrderRow, SeriesOrdering


@dataclass(frozen=True, slots=True)
class EpisodeOrder:
    """Another TVDB order of the series, read onto Sonarr's episodes through the TVDB episode ids.

    An episode only that order has (a split or a merged one) has a key but no id.
    """

    id_by_key: Mapping[EpisodeKey, int]
    """`(season, episode)` in the order -> Sonarr's episode id, for the episodes Sonarr has."""

    keys: frozenset[EpisodeKey]
    """Every `(season, episode)` the order numbers, Sonarr episode or not."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id_by_key", MappingProxyType(dict(self.id_by_key)))


@dataclass(frozen=True, slots=True)
class EpisodeIndex:
    """The import family's one episode index, built once per episode fetch.

    The lookups are derived at construction, and `titles` builds each title view on first use. Nothing under the
    index changes after it is built, and every torrent and window placed under it shares them.

    The three facets are copied and made read-only at construction. `episode_index` is the one builder. Episodes
    with a falsy id (0) are dropped from EVERY facet before keying: a 0 id can never be POSTed to Sonarr, and a
    real-id twin behind one must still win its `(season, episode)` key. The planner's `index_episodes_by_key`
    deliberately KEEPS them (a 0-id file is still identity evidence). Do not unify the two.
    """

    by_id: Mapping[int, SonarrEpisode]
    """Episode id -> episode, in the fetch's (season) order. `list(by_id)` is
    the scope id list the grab stores on each entry's claim."""

    id_by_key: Mapping[EpisodeKey, int]
    """`(season, episode)` -> episode id (missing numbers collapse to
    `SONARR_MISSING_KEY`, first record wins, via `index_episodes_by_key`)."""

    orders: Mapping[str, EpisodeOrder] = field(default_factory=dict[str, EpisodeOrder])
    """The series' other TVDB orders, by name, each read onto these episodes. Never the official order."""

    key_by_id: Mapping[int, EpisodeKey] = field(init=False, repr=False, compare=False)
    """Episode id -> `(season, episode)`."""
    season_counts: Mapping[int, int] = field(init=False, repr=False, compare=False)
    """Season -> how many episodes it has."""
    absolute_of: Mapping[int, int] = field(init=False, repr=False, compare=False)
    """Episode id -> absolute number, for the episodes that have one."""
    id_by_absolute: Mapping[int, int] = field(init=False, repr=False, compare=False)
    """Absolute number -> episode id, the inverse of `absolute_of` (the last id wins a repeated number)."""
    _titles: dict[frozenset[str], EpisodeTitles] = field(
        default_factory=dict[frozenset[str], EpisodeTitles], init=False, repr=False, compare=False
    )
    """The memo `titles` fills: one `EpisodeTitles` per set of series title words placed under this index."""

    def __post_init__(self) -> None:
        # Detach from the caller's dicts, then wrap read-only.
        object.__setattr__(self, "by_id", MappingProxyType(dict(self.by_id)))
        object.__setattr__(self, "id_by_key", MappingProxyType(dict(self.id_by_key)))
        object.__setattr__(self, "orders", MappingProxyType(dict(self.orders)))
        keys = self.id_by_key
        object.__setattr__(self, "key_by_id", MappingProxyType({ep_id: key for key, ep_id in keys.items()}))
        object.__setattr__(self, "season_counts", MappingProxyType(dict(Counter(key.season for key in keys))))
        absolute_of = {
            ep_id: ep.absolute_episode_number
            for ep_id, ep in self.by_id.items()
            if ep.absolute_episode_number is not None
        }
        object.__setattr__(self, "absolute_of", MappingProxyType(absolute_of))
        object.__setattr__(
            self, "id_by_absolute", MappingProxyType({number: ep_id for ep_id, number in absolute_of.items()})
        )

    def titles(self, series_words: frozenset[str]) -> EpisodeTitles:
        """The episode titles as words beyond the series title's words, built once per word set under this index."""

        titles = self._titles.get(series_words)
        if titles is None:
            titles = self._titles[series_words] = EpisodeTitles.of(self.by_id, self.id_by_key, series_words)
        return titles


def episode_index(ep_list: Iterable[SonarrEpisode], ordering: SeriesOrdering = NO_ORDERING) -> EpisodeIndex:
    """Fold one `/api/v3/episode` fetch into the `EpisodeIndex` facets, with `ordering.others` read onto it."""

    with_ids = [ep for ep in ep_list if ep.id]
    id_by_tvdb = {ep.tvdb_id: ep.id for ep in with_ids if ep.tvdb_id}
    return EpisodeIndex(
        by_id={ep.id: ep for ep in with_ids},
        id_by_key={key: ep.id for key, ep in index_episodes_by_key(with_ids).items()},
        orders={name: _read_order(rows, id_by_tvdb) for name, rows in ordering.others.items()},
    )


def _read_order(rows: Iterable[OrderRow], id_by_tvdb: Mapping[int, int]) -> EpisodeOrder:
    """The order read onto Sonarr's episodes: the first row of a repeated key wins."""

    id_by_key: dict[EpisodeKey, int] = {}
    keys: set[EpisodeKey] = set()
    for row in rows:
        key = EpisodeKey(row.season, row.episode)
        keys.add(key)
        if (ep_id := id_by_tvdb.get(row.tvdb_id)) is not None:
            id_by_key.setdefault(key, ep_id)
    return EpisodeOrder(id_by_key, frozenset(keys))


def all_specials(series: EpisodeIndex, ids: Iterable[int]) -> bool:
    """Whether every id is one of the series' specials (season 0)."""

    return all((ep := series.by_id.get(ep_id)) is not None and ep.season_number == 0 for ep_id in ids)


class PlacementVerdict(StrEnum):
    """How one file was judged: placed by which pass, excluded, or unplaced.

    `assign_episode_ids` gives every verdict but two, which are given once the windows merge: `NUMBERINGS_AGREE`
    and `LISTED_ELSEWHERE`.
    """

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
    OTHER_ORDER = "another TVDB order"
    """Onto the episode its own number names in another TVDB order of the series (DVD, alternate, and so on) that
    explains the torrent's numbers where the aired order fails on them (`placer._orders`)."""
    NUMBERINGS_AGREE = "every numbering agrees"
    """A file named by number alone that no window placed or refused, placed where its numbers fall once every way
    the group could have counted them agrees (`numberings`). Given after the windows merge, never by a pass."""
    ALIASED_ELSEWHERE = "aliased special outside the window"
    """Not placed here: a file of a misnumbered specials pack whose alias points at a special outside this window.
    A claim, not a refusal, so the window that holds that special may still place it."""
    FOREIGN = "other slice"
    """Reads wholly outside the scope, or is titled as an episode outside it and confirmed by its reading, by that
    reading's rejection while other files hold its episodes, or by no open id being left: not this record's to
    import under its current claims."""
    LISTED_ELSEWHERE = "another listed title's"
    """Placed by the window of another title SeaDex lists the torrent under, one with no claim on the record, or
    left for a hand import by such a window with none of the record's claimed windows party to it: not this
    record's to import or hand-import under its current claims. That title's own grab claims the torrent and places
    or reports the file, unless the title is in `ignore_anilist_ids`, which is never grabbed for."""
    DUPLICATE = "duplicate"
    """Reads inside the scope onto episodes other files provably hold, or is titled as an episode a version of the
    same file holds."""
    EXTRA = "extra"
    """An opening, an ending, a preview, a menu, a commercial, a trailer, a teaser, or a promo: never an episode."""
    HELD = "held"
    """A file of the picked release run while a parse is unknown: no other pass may place it, and it is re-checked."""
    MISNUMBERED = "misnumbered"
    """A file whose own numbers disagree, or whose count no title confirms, that no run places. Or a file of a
    specials pack whose numbers don't match the windows listing it, when its special aliases can't place it either
    or another window places it elsewhere. Never placed by its numbers, left for a hand import."""
    TITLE_REFUTED = "refuted by its run's titles"
    """A plain member of a numbered run whose titled members contradict the run's numbers, when nothing else placed
    it. A claim, not a refusal: another window may still place it. Left for a hand import when none does."""
    OTHER_ORDER_ELSEWHERE = "another TVDB order, outside the window"
    """Not placed here: a misnumbered or refuted file that the TVDB order explaining the torrent reads onto an
    episode outside this window. A claim, not a refusal, so the window that holds the episode may place it. Left for
    a hand import when none does."""
    ORDER_ONLY = "another TVDB order, not in Sonarr's list"
    """A file whose number the TVDB order explaining the torrent gives to an episode Sonarr doesn't list (one half
    of an episode that order splits in two, say). Not placed under its window, and left for a hand import unless
    another window places it or proves it a duplicate. `Placement.order_only` names the order and the key."""
    SKIPPED = "skipped"
    """Nothing placed it and nothing proved it foreign."""

    @property
    def placed(self) -> bool:
        """Whether the verdict carries episode ids."""

        return isinstance(_KIND[self], _Places)

    @property
    def excluded(self) -> bool:
        """Whether the file isn't this record's to import under its current claims.

        A later claim's window can still place a foreign file or another listed title's.
        """

        row = _KIND[self]
        return not isinstance(row, _Places) and row.leaves is _Unplaced.EXCLUDED

    @property
    def refused(self) -> bool:
        """Whether the verdict settles the file with no ids: held for a re-ask, or misnumbered for a hand import."""

        return isinstance(_KIND[self], _Refuses)

    @property
    def hand_import(self) -> bool:
        """Whether the verdict leaves the file for a hand import.

        Misnumbered, refuted by its run's titles, or read by another TVDB order onto an episode outside the window
        or one Sonarr doesn't list.
        """

        row = _KIND[self]
        return not isinstance(row, _Places) and row.leaves is _Unplaced.HAND_IMPORT

    @property
    def decides(self) -> bool:
        """Whether a window that gives this verdict decides the file: it placed or refused it."""

        return self.placed or self.refused

    @property
    def by_alias(self) -> bool:
        """Whether the special aliases gave the verdict.

        Placed on the aliased special, or claimed for the window that holds it.
        """

        row = _KIND[self]
        return isinstance(row, _Places | _Claims) and row.by_alias

    @property
    def party(self) -> bool:
        """Whether a window giving this verdict is party to the file.

        It placed or refused the file, left it for a hand import, or claimed it by alias. A window that wrote the file
        off (another slice's, a duplicate, an extra, a skip) isn't.
        """

        return self.decides or self.by_alias or self.hand_import

    @property
    def claim_rank(self) -> int:
        """How strong a claim the verdict is: a file no window decides ends with its highest-ranked claim.

        Read it only on a claim, a verdict a window gives without deciding the file.
        """

        row = _KIND[self]
        assert isinstance(row, _Claims)
        return row.rank

    @property
    def left_open(self) -> bool:
        """Whether another TVDB order (pass 16) may still place the file.

        Nothing placed it, and nothing proved it a duplicate or an extra.
        """

        row = _KIND[self]
        return isinstance(row, _Refuses | _Claims) and row.left_open


class _Unplaced(Enum):
    """What a verdict that carries no ids means for the record."""

    EXCLUDED = auto()
    """The file isn't this record's to import under its current claims."""
    HAND_IMPORT = auto()
    """The file is left for a hand import."""
    STAYS = auto()
    """The file stays with the record as one nothing placed, and the caller warns of it."""


@dataclass(frozen=True, slots=True)
class _Places:
    """The verdict carries ids. A window that gives it decides the file."""

    by_alias: bool = False
    """Whether the special aliases placed it."""


@dataclass(frozen=True, slots=True)
class _Refuses:
    """The verdict settles the file with no ids. A window that gives it decides the file."""

    leaves: _Unplaced
    left_open: bool = False
    """Whether another TVDB order may still place the file (pass 16)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Claims:
    """One window's view of a file it neither placed nor refused. A later window may still decide the file."""

    leaves: _Unplaced
    rank: int
    """A file no window decides ends with its highest-ranked claim."""
    by_alias: bool = False
    """Whether the special aliases claimed it for another window."""
    left_open: bool = False
    """Whether another TVDB order may still place the file (pass 16)."""


@dataclass(frozen=True, slots=True)
class _AfterMerge:
    """No window gives the verdict: it is set once the windows have merged."""

    leaves: _Unplaced


type _Kind = _Places | _Refuses | _Claims | _AfterMerge

_KIND: Mapping[PlacementVerdict, _Kind] = MappingProxyType(
    {
        PlacementVerdict.EXACT: _Places(),
        PlacementVerdict.RELEASE_RUN: _Places(),
        PlacementVerdict.ABSOLUTE: _Places(),
        PlacementVerdict.SINGLE: _Places(),
        PlacementVerdict.ORDERED: _Places(),
        PlacementVerdict.NUMBERED_RUN: _Places(),
        PlacementVerdict.TITLED: _Places(),
        PlacementVerdict.EPISODE_TITLE: _Places(),
        PlacementVerdict.IDENTIFIED: _Places(),
        PlacementVerdict.OVERRIDDEN: _Places(),
        PlacementVerdict.ALTERNATE: _Places(by_alias=True),
        PlacementVerdict.OTHER_ORDER: _Places(),
        PlacementVerdict.NUMBERINGS_AGREE: _Places(),
        PlacementVerdict.HELD: _Refuses(_Unplaced.STAYS),
        PlacementVerdict.MISNUMBERED: _Refuses(_Unplaced.HAND_IMPORT, left_open=True),
        # The claims, weakest first. EXTRA is lowest because it rests on one window's titles. SKIPPED beats FOREIGN
        # and EXTRA, so a file some window couldn't place is never written off, and ALIASED_ELSEWHERE beats FOREIGN
        # because the aliases name the file as this series' special. ORDER_ONLY beats SKIPPED, so a file one
        # window's order reads as no episode of Sonarr's stays a hand import where another window only skipped it,
        # and a window that proves it a duplicate still wins. OTHER_ORDER_ELSEWHERE and TITLE_REFUTED are highest,
        # so a file whose numbers a window refused or refuted stays a hand import.
        PlacementVerdict.EXTRA: _Claims(leaves=_Unplaced.EXCLUDED, rank=0),
        PlacementVerdict.FOREIGN: _Claims(leaves=_Unplaced.EXCLUDED, rank=1, left_open=True),
        PlacementVerdict.ALIASED_ELSEWHERE: _Claims(leaves=_Unplaced.STAYS, rank=2, by_alias=True),
        PlacementVerdict.SKIPPED: _Claims(leaves=_Unplaced.STAYS, rank=3, left_open=True),
        PlacementVerdict.ORDER_ONLY: _Claims(leaves=_Unplaced.HAND_IMPORT, rank=4),
        PlacementVerdict.DUPLICATE: _Claims(leaves=_Unplaced.EXCLUDED, rank=5),
        PlacementVerdict.OTHER_ORDER_ELSEWHERE: _Claims(leaves=_Unplaced.HAND_IMPORT, rank=6),
        PlacementVerdict.TITLE_REFUTED: _Claims(leaves=_Unplaced.HAND_IMPORT, rank=7, left_open=True),
        PlacementVerdict.LISTED_ELSEWHERE: _AfterMerge(_Unplaced.EXCLUDED),
    }
)
"""What each verdict is: one row per verdict, of one of four kinds."""


@dataclass(frozen=True, slots=True)
class Placement:
    """One file's verdict, with its episode ids when placed."""

    name: str
    """The normalized basename."""
    ids: tuple[int, ...]
    """The episode ids, in claim order (empty unless `verdict.placed`)."""
    verdict: PlacementVerdict
    """How the file left the passes."""
    order_only: OrderOnlyKey | None = None
    """The order and key that read the file, when the verdict is `ORDER_ONLY`."""
    read_onto: tuple[int, ...] = ()
    """The episodes the order reads the file onto, when the verdict is `OTHER_ORDER_ELSEWHERE`."""


@dataclass(frozen=True, slots=True)
class EpisodeAssignment:
    """The outcome of assigning a torrent's on-disk files to resolved episode ids, one verdict per file."""

    placements: tuple[Placement, ...]
    """One per name in `PlacementBatch.names`, batch order."""

    @property
    def assigned(self) -> dict[str, tuple[int, ...]]:
        """Normalized basename -> episode ids for every placed file (each id in the scope, used once)."""

        return {p.name: p.ids for p in self.placements if p.verdict.placed}

    @property
    def skipped(self) -> tuple[str, ...]:
        """The files nothing placed and nothing excluded: the caller warns, never guesses."""

        return tuple(p.name for p in self.placements if not p.verdict.placed and not p.verdict.excluded)

    @property
    def excluded(self) -> tuple[Placement, ...]:
        """The files not this record's under its current claims, and how (`PlacementVerdict.excluded`)."""

        return tuple(p for p in self.placements if p.verdict.excluded)

    @property
    def hand_import_files(self) -> tuple[str, ...]:
        """The skipped files left for a hand import (`PlacementVerdict.hand_import`)."""

        return tuple(p.name for p in self.placements if p.verdict.hand_import)

    @property
    def order_only(self) -> dict[str, OrderOnlyKey]:
        """Normalized basename -> the order and key that read it, for every `ORDER_ONLY` file."""

        return {p.name: p.order_only for p in self.placements if p.order_only is not None}


NO_SEEDED: FileEpisodeMap = MappingProxyType({})
"""No stored map."""


@dataclass(frozen=True, slots=True)
class PlacementBatch:
    """A torrent's files to place, with the parses of the WHOLE torrent and its stored map.

    `parsed` may cover MORE files than `to_place`. Every parsed file is read as evidence, placed or not, and a name
    `seeded` holds is never placed, in `to_place` or not.
    """

    to_place: Sequence[str]
    """Normalized basenames in SeaDex order (the order only fixes deterministic output), held as a tuple."""

    parsed: Mapping[str, ParsedFileInfo | None]
    """Series-agnostic parse per file (None when Sonarr's parse was unavailable)."""

    seeded: FileEpisodeMap = NO_SEEDED
    """The stored record's map, normalized name -> episode ids, the ids held as tuples. Its ids count as used wherever
    the scope admits them, and the misnumbered pack pass checks each seeded member sits on its aliased special."""

    names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    """The distinct names to place, batch order: `to_place` less the names the map holds."""

    torrent_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    """The distinct names the placer reads: the names to place, then the rest of `parsed` (seeded, gone). A mapped
    name with no parse entry is left out."""

    all_parses_known: bool = field(init=False, repr=False, compare=False)
    """Every parse came from Sonarr this run: no miss (None). Seeded and gone files are parsed by name too, so a miss
    on any of them counts, as does a file to place that has no parse entry."""

    def __post_init__(self) -> None:
        # Detach from the caller's collections, then wrap read-only. The ids are tupled whatever the type says: a
        # cast or an untyped script can hand lists, and a list never equals `(ep_id,)` in `_aliased_specials`, so a
        # seed on its aliased special would read as off it and the misnumbered pack would lose its aliases.
        object.__setattr__(self, "to_place", tuple(self.to_place))
        object.__setattr__(self, "parsed", MappingProxyType(dict(self.parsed)))
        object.__setattr__(self, "seeded", MappingProxyType({name: tuple(ids) for name, ids in self.seeded.items()}))
        names = tuple(name for name in dict.fromkeys(self.to_place) if name not in self.seeded)
        torrent_names = tuple(dict.fromkeys([*names, *self.parsed]))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "torrent_names", torrent_names)
        object.__setattr__(self, "all_parses_known", all(self.parsed.get(name) is not None for name in torrent_names))


type SizeIdentities = Mapping[int, int]
"""File size -> episode id, from SeaDex releases with a single video file listed under an entry that covers one
episode. A file with that exact size is that episode."""


@dataclass(frozen=True, slots=True)
class TorrentListing:
    """The windows of the SeaDex entries that list one torrent, and those entries' special aliases."""

    entries: tuple[EntryWindow, ...] = ()
    """One window per listing entry, in AniList id order."""

    special_aliases: SpecialAliases = NO_ALIASES
    """The union of those entries' special aliases, minus any TMDB number two entries pair differently. Empty when
    the entries pair against different TMDB shows."""

    ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """Union of the entries' episode ids."""

    def __post_init__(self) -> None:
        # SeaDex can list one hash twice under an entry, so the first window per AniList id wins.
        kept: dict[int, EntryWindow] = {}
        for window in self.entries:
            kept.setdefault(window.al_id, window)
        entries = tuple(sorted(kept.values(), key=lambda window: window.al_id))
        object.__setattr__(self, "entries", entries)
        # Copied before it's wrapped, so a caller's later edits can't reach the listing.
        object.__setattr__(self, "special_aliases", MappingProxyType(dict(self.special_aliases)))
        object.__setattr__(self, "ids", frozenset(ep_id for window in entries for ep_id in window.ordered_episode_ids))


EMPTY_LISTING = TorrentListing()
"""A torrent no entry lists. Listings match by infohash, so a url without one always gets this."""

type IdentifiedNames = Mapping[str, int]
"""Normalized file name -> the episode id its size matched."""

NO_IDENTIFIED: IdentifiedNames = MappingProxyType({})
"""No files placed by size: no size matched, or the size identities weren't read."""


@dataclass(frozen=True, slots=True)
class ListingEvidence:
    """What SeaDex's listings for the series tell the placer about one torrent, as this run read them."""

    listing: TorrentListing = EMPTY_LISTING
    """The torrent's listing: the ids a numbered specials pack is checked against, and the special aliases that
    can place a misnumbered one. Empty when no entry lists the torrent, which turns the pack check off."""

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


@dataclass(frozen=True, slots=True)
class TargetScope:
    """The episode set a placement batch may assign into.

    `resolved` is the FULL id list and `used` the ids already held: an earlier window's, plus the stored map's the
    placer adds (`Placer.scope`). The two stay separate so a fully seeded record still enforces its scope, while an
    EMPTY `resolved` means no scope at all (`unscoped`): the exact pass then places a name's own keys against the
    whole series map instead of placing nothing forever.
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
    """Ids already held, by an earlier window or the stored map. Never placed again."""

    names: EntryNames = field(default_factory=EntryNames)
    """The series and AniList titles. Episode titles and extras words are read against them, and they break
    ties when several runs or numberless files could fill the open ids."""

    listing: ListingEvidence = NO_EVIDENCE
    """What SeaDex's listings tell us about this torrent this run."""

    earlier: FileEpisodeMap = field(default_factory=dict[str, tuple[int, ...]])
    """Normalized name -> the ids an earlier window placed it on, whether or not this window admits them (`after`).
    The other-orders pass reads them as settled."""

    episode_count: int | None = None
    """AniList's episode count for the title, None when the window carries none (`EntryWindow.episode_count`)."""

    real_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """The resolved ids that can be placed. A stray zero keeps the scope real but is never one."""

    def __post_init__(self) -> None:
        # Detached from the caller's list: `real_ids` is derived from the list once.
        object.__setattr__(self, "resolved", tuple(self.resolved))
        object.__setattr__(self, "real_ids", frozenset(i for i in self.resolved if i))
        object.__setattr__(self, "earlier", MappingProxyType(dict(self.earlier)))

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

    def after(self, earlier: FileEpisodeMap) -> Self:
        """The scope of a window placed after others: `earlier` are their placements, the ids used where admitted."""

        return replace(self.using(ep_id for ids in earlier.values() for ep_id in ids), earlier=earlier)

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


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedScope:
    """What one entry's files are placed against: its indexes, its names, and what AniList and Sonarr say of it."""

    al_id: int
    """The entry claiming the placement: a stored claim of its own is replaced by the fresh window."""

    series_id: int
    """The Sonarr series the entry's episodes belong to."""

    entry: EpisodeIndex
    """The entry's episodes: the resolved set, and what the slice and preowned reads cover."""

    series: EpisodeIndex
    """The WHOLE series (empty when the series list could not be read, which places nothing)."""

    names: EntryNames
    """The series and AniList titles the placement breaks ties by, persisted onto every seed."""

    episode_count: int | None
    """AniList's episode count for the entry (`SonarrEpisodes.episode_count`), None when AniList gives none."""

    alternate_titles: tuple[str, ...]
    """The series' other titles in Sonarr (`SonarrSeries.alternateTitles`), in Sonarr's order."""

    aired_numbering: bool
    """Whether Sonarr numbers the series by TVDB's aired order (`SonarrEpisodes.numbers_aired`)."""

    window: EntryWindow = field(init=False, repr=False, compare=False)
    """The entry's own window, built as a listing builds an entry's (`EntryWindow.of`)."""

    def __post_init__(self) -> None:
        window = EntryWindow.of(
            al_id=self.al_id,
            series_id=self.series_id,
            ep_list=self.entry.by_id.values(),
            names=self.names,
            episode_count=self.episode_count,
        )
        object.__setattr__(self, "window", window)

    @property
    def can_place(self) -> bool:
        """Whether there is an entry index and a served series map to place against.

        An empty entry index must not pass as "no scope" (which places against the whole series map), and an
        unread map places nothing: a grab-time verdict is final.
        """

        return bool(self.entry.by_id and self.series.id_by_key)


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
