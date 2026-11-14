"""Types the placement modules share: the episode index, verdicts, one file's placement, the batch, and the targets."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Self

from .anibridge import NO_ALIASES, SpecialAliases
from .arr_models import ParsedFileInfo, SonarrEpisode, index_episodes_by_key
from .episode_titles import EpisodeTitles
from .pending_import import FileEpisodeMap, TitleNames, TitleWindow
from .releases import EpisodeKey, OrderOnlyKey
from .tvdb_orders import NO_INDEX_ORDERS, IndexOrders, OrderRow


@dataclass(frozen=True, slots=True)
class TvdbOrder:
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
    the window id list the grab stores on each title's claim."""

    id_by_key: Mapping[EpisodeKey, int]
    """`(season, episode)` -> episode id (missing numbers collapse to
    `SONARR_MISSING_KEY`, first episode wins, via `index_episodes_by_key`)."""

    orders: Mapping[str, TvdbOrder] = field(default_factory=dict[str, TvdbOrder])
    """The series' other TVDB orders, by name, each read onto these episodes. Never the aired order."""

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
    """The memo `titles` fills: one `EpisodeTitles` per set of series name words placed under this index."""

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
        """The episode titles as words beyond the series name words, built once per word set under this index."""

        titles = self._titles.get(series_words)
        if titles is None:
            titles = self._titles[series_words] = EpisodeTitles.of(self.by_id, self.id_by_key, series_words)
        return titles


def episode_index(ep_list: Iterable[SonarrEpisode], ordering: IndexOrders = NO_INDEX_ORDERS) -> EpisodeIndex:
    """Convert one `/api/v3/episode` fetch into the `EpisodeIndex` facets, with `ordering.others` read onto it."""

    with_ids = [ep for ep in ep_list if ep.id]
    id_by_tvdb = {ep.tvdb_id: ep.id for ep in with_ids if ep.tvdb_id}
    return EpisodeIndex(
        by_id={ep.id: ep for ep in with_ids},
        id_by_key={key: ep.id for key, ep in index_episodes_by_key(with_ids).items()},
        orders={name: _read_order(rows, id_by_tvdb) for name, rows in ordering.others.items()},
    )


def _read_order(rows: Iterable[OrderRow], id_by_tvdb: Mapping[int, int]) -> TvdbOrder:
    """The order read onto Sonarr's episodes: the first row of a repeated key wins."""

    id_by_key: dict[EpisodeKey, int] = {}
    keys: set[EpisodeKey] = set()
    for row in rows:
        key = EpisodeKey(row.season, row.episode)
        keys.add(key)
        if (ep_id := id_by_tvdb.get(row.tvdb_id)) is not None:
            id_by_key.setdefault(key, ep_id)
    return TvdbOrder(id_by_key, frozenset(keys))


def all_specials(series: EpisodeIndex, ids: Iterable[int]) -> bool:
    """Whether every id is one of the series' specials (season 0)."""

    return all((ep := series.by_id.get(ep_id)) is not None and ep.season_number == 0 for ep_id in ids)


class PlacementVerdict(StrEnum):
    """How one file was judged: placed by which pass, excluded, or unplaced.

    `place_batch` gives every verdict but two, which are given once the windows merge: `NUMBERINGS_AGREE`
    and `ANOTHER_TITLES`.
    """

    EXACT = "exact"
    """Its parsed season and episode, or Sonarr's matched episode, resolved inside the window."""
    SEQUENCE_OVER_PARSE = "release run"
    """A numbered sequence's own numbers placed it onto the consecutive unused episodes, overriding Sonarr's
    inconsistent parse."""
    ABSOLUTE = "absolute"
    """Paired with an unused id in absolute-number order (pairing in absolute order)."""
    SINGLE = "single file"
    """The only candidate file, with no episode number, onto the only unused id."""
    NAME_ORDER = "ordered"
    """A batch of files with no parsed number that no earlier pass touched, paired with the unused ids in natural
    name order."""
    SEQUENCE = "numbered run"
    """A `1..N` sequence among the files Sonarr found no episode for, paired with the consecutive unused episodes."""
    TITLE_NAME = "titled"
    """The file with no parsed number that an AniList name of the title points at, onto the only unused id."""
    EPISODE_TITLE = "episode title"
    """Onto the one episode whose title its name carries, whatever its number said."""
    SIZE_MATCH = "listed size"
    """Placed by file size: SeaDex lists a single-file release of this size for the episode, and the name agrees or
    has no number."""
    SIZE_OVER_NAME = "listed size over its name"
    """Placed by file size even though the name's number points at a different episode. The caller logs a warning."""
    SPECIAL_ALIAS = "alternate numbering"
    """A file of a batch of misnumbered specials, placed on the special its number's alias points at."""
    OTHER_ORDER = "another TVDB order"
    """Onto the episode its own number names in another TVDB order of the series (DVD, alternate, and so on) that
    explains the torrent's numbers where the aired order fails on them (`placer._orders`)."""
    NUMBERINGS_AGREE = "every numbering agrees"
    """A file named by number alone that no window gave a final verdict, placed where its numbers fall once every
    way the group could have numbered them agrees (`numberings`). Given after the windows merge, never by a pass."""
    ALIAS_OUTSIDE_WINDOW = "aliased special outside the window"
    """Not placed here: a file of a batch of misnumbered specials whose alias points at a special outside this
    window. A provisional verdict, so the window that holds that special may still place it."""
    OUTSIDE_WINDOW = "other slice"
    """Parses wholly outside the window, or carries the title of an episode outside it, confirmed by its parse, by
    that parse's rejection while its episodes are used, or by no unused id being left: not this pending import's to
    import under its current claims."""
    ANOTHER_TITLES = "another listed title's"
    """Placed by the window of another title SeaDex lists the torrent under, one with no claim on the pending
    import, or left for a hand import by such a window with none of the pending import's claimed windows involved
    in it: not this pending import's to import or hand-import under its current claims. That title's own grab
    claims the torrent and places or reports the file, unless the title is in `ignore_anilist_ids`, which is never
    grabbed for."""
    DUPLICATE = "duplicate"
    """Parses inside the window onto episodes other files are provably placed on, or carries the title of an
    episode a version of the same file is placed on."""
    EXTRA = "extra"
    """An opening, an ending, a preview, a menu, a commercial, a trailer, a teaser, or a promo: never an episode."""
    WAITING_ON_PARSE = "held"
    """A file of the sequence the sequence-over-parse pass picked, while a parse is unknown: no other pass may
    place it, and it is re-checked."""
    MISNUMBERED = "misnumbered"
    """A file whose own numbers disagree, or whose number no title confirms, that no sequence places. Or a file of
    a batch of specials whose numbers don't match the titles listing it, when its special aliases can't place it
    either or another window places it elsewhere. Never placed by its numbers, left for a hand import."""
    EPISODE_TITLES_CONTRADICT = "refuted by its run's titles"
    """A plain member of a numbered sequence whose members carrying titles contradict the sequence's numbers, when
    nothing else placed it. A provisional verdict: another window may still place it. Left for a hand import when
    none does."""
    OTHER_ORDER_OUTSIDE_WINDOW = "another TVDB order, outside the window"
    """Not placed here: a misnumbered or contradicted file that the TVDB order explaining the torrent reads onto an
    episode outside this window. A provisional verdict, so the window that holds the episode may place it. Left for
    a hand import when none does."""
    ORDER_ONLY = "another TVDB order, not in Sonarr's list"
    """A file whose number the TVDB order explaining the torrent gives to an episode Sonarr doesn't list (one half
    of an episode that order splits in two, say). Not placed under its window, and left for a hand import unless
    another window places it or proves it a duplicate. `Placement.order_only` names the order and the key."""
    NO_EPISODE_FOUND = "skipped"
    """Nothing placed it and nothing proved it outside the window."""

    @property
    def placed(self) -> bool:
        """Whether the verdict carries episode ids."""

        return isinstance(_KIND[self], _Placed)

    @property
    def excluded(self) -> bool:
        """Whether the file isn't this pending import's to import under its current claims.

        A later claim's window can still place a file outside the window, or another title's.
        """

        row = _KIND[self]
        return not isinstance(row, _Placed) and row.leaves is _Leaves.EXCLUDED

    @property
    def final_unplaced(self) -> bool:
        """Whether the verdict is final with no ids: waiting on a parse, or misnumbered for a hand import."""

        return isinstance(_KIND[self], _FinalUnplaced)

    @property
    def left_for_hand_import(self) -> bool:
        """Whether the verdict leaves the file for a hand import.

        Misnumbered, contradicted by its sequence's titles, or read by another TVDB order onto an episode outside the
        window or one Sonarr doesn't list.
        """

        row = _KIND[self]
        return not isinstance(row, _Placed) and row.leaves is _Leaves.HAND_IMPORT

    @property
    def final(self) -> bool:
        """Whether the verdict is final: placed, or final with no ids (`final_unplaced`)."""

        return self.placed or self.final_unplaced

    @property
    def from_alias(self) -> bool:
        """Whether the special aliases gave the verdict.

        Placed on the aliased special, or left provisional for the window that holds it.
        """

        row = _KIND[self]
        return isinstance(row, _Placed | _Provisional) and row.from_alias

    @property
    def involves(self) -> bool:
        """Whether a window giving this verdict is involved in the file.

        It gave a final verdict, left the file for a hand import, or gave it `ALIAS_OUTSIDE_WINDOW`. A window that
        wrote the file off (outside the window, a duplicate, an extra, no episode found) isn't.
        """

        return self.final or self.from_alias or self.left_for_hand_import

    @property
    def rank(self) -> int:
        """How strong a provisional verdict is: a file no window gives a final verdict ends with its highest-ranked one.

        Read it only on a provisional verdict.
        """

        row = _KIND[self]
        assert isinstance(row, _Provisional)
        return row.rank

    @property
    def other_order_may_place(self) -> bool:
        """Whether another TVDB order (pass 16) may still place the file.

        Nothing placed it, and nothing proved it a duplicate or an extra.
        """

        row = _KIND[self]
        return isinstance(row, _FinalUnplaced | _Provisional) and row.other_order_may_place


class _Leaves(Enum):
    """What a verdict that carries no ids means for the pending import."""

    EXCLUDED = auto()
    """The file isn't this pending import's to import under its current claims."""
    HAND_IMPORT = auto()
    """The file is left for a hand import."""
    STAYS = auto()
    """The file stays with the pending import as one nothing placed, and the caller warns of it."""


@dataclass(frozen=True, slots=True)
class _Placed:
    """The verdict carries ids, and it is final."""

    from_alias: bool = False
    """Whether the special aliases placed it."""


@dataclass(frozen=True, slots=True)
class _FinalUnplaced:
    """The verdict is final and carries no ids."""

    leaves: _Leaves
    other_order_may_place: bool = False
    """Whether another TVDB order may still place the file (pass 16)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Provisional:
    """One window's view of a file it gave no final verdict. A later window may still give one."""

    leaves: _Leaves
    rank: int
    """A file no window gives a final verdict ends with its highest-ranked provisional one."""
    from_alias: bool = False
    """Whether the special aliases left it for another window."""
    other_order_may_place: bool = False
    """Whether another TVDB order may still place the file (pass 16)."""


@dataclass(frozen=True, slots=True)
class _AfterMerge:
    """No window gives the verdict: it is set once the windows have merged."""

    leaves: _Leaves


type _Kind = _Placed | _FinalUnplaced | _Provisional | _AfterMerge

_KIND: Mapping[PlacementVerdict, _Kind] = MappingProxyType(
    {
        PlacementVerdict.EXACT: _Placed(),
        PlacementVerdict.SEQUENCE_OVER_PARSE: _Placed(),
        PlacementVerdict.ABSOLUTE: _Placed(),
        PlacementVerdict.SINGLE: _Placed(),
        PlacementVerdict.NAME_ORDER: _Placed(),
        PlacementVerdict.SEQUENCE: _Placed(),
        PlacementVerdict.TITLE_NAME: _Placed(),
        PlacementVerdict.EPISODE_TITLE: _Placed(),
        PlacementVerdict.SIZE_MATCH: _Placed(),
        PlacementVerdict.SIZE_OVER_NAME: _Placed(),
        PlacementVerdict.SPECIAL_ALIAS: _Placed(from_alias=True),
        PlacementVerdict.OTHER_ORDER: _Placed(),
        PlacementVerdict.NUMBERINGS_AGREE: _Placed(),
        PlacementVerdict.WAITING_ON_PARSE: _FinalUnplaced(_Leaves.STAYS),
        PlacementVerdict.MISNUMBERED: _FinalUnplaced(_Leaves.HAND_IMPORT, other_order_may_place=True),
        # The provisional verdicts, weakest first. EXTRA is lowest because it rests on one window's titles.
        # NO_EPISODE_FOUND beats OUTSIDE_WINDOW and EXTRA, so a file some window couldn't place is never written off,
        # and ALIAS_OUTSIDE_WINDOW beats OUTSIDE_WINDOW because the aliases name the file as this series' special.
        # ORDER_ONLY beats NO_EPISODE_FOUND, so a file one window's order reads as no episode of Sonarr's stays a hand
        # import where another window found no episode for it, and a window that proves it a duplicate still wins.
        # OTHER_ORDER_OUTSIDE_WINDOW and EPISODE_TITLES_CONTRADICT are highest, so a file whose numbers a window
        # rejected or a title contradicted stays a hand import.
        PlacementVerdict.EXTRA: _Provisional(leaves=_Leaves.EXCLUDED, rank=0),
        PlacementVerdict.OUTSIDE_WINDOW: _Provisional(leaves=_Leaves.EXCLUDED, rank=1, other_order_may_place=True),
        PlacementVerdict.ALIAS_OUTSIDE_WINDOW: _Provisional(leaves=_Leaves.STAYS, rank=2, from_alias=True),
        PlacementVerdict.NO_EPISODE_FOUND: _Provisional(leaves=_Leaves.STAYS, rank=3, other_order_may_place=True),
        PlacementVerdict.ORDER_ONLY: _Provisional(leaves=_Leaves.HAND_IMPORT, rank=4),
        PlacementVerdict.DUPLICATE: _Provisional(leaves=_Leaves.EXCLUDED, rank=5),
        PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW: _Provisional(leaves=_Leaves.HAND_IMPORT, rank=6),
        PlacementVerdict.EPISODE_TITLES_CONTRADICT: _Provisional(
            leaves=_Leaves.HAND_IMPORT, rank=7, other_order_may_place=True
        ),
        PlacementVerdict.ANOTHER_TITLES: _AfterMerge(_Leaves.EXCLUDED),
    }
)
"""What each verdict is: one row per verdict, of one of four kinds."""


@dataclass(frozen=True, slots=True)
class Placement:
    """One file's verdict, with its episode ids when placed."""

    name: str
    """The normalized file name."""
    ids: tuple[int, ...]
    """The episode ids, in the order placed (empty unless `verdict.placed`)."""
    verdict: PlacementVerdict
    """How the file left the passes."""
    order_only: OrderOnlyKey | None = None
    """The order and key that read the file, when the verdict is `ORDER_ONLY`."""
    other_order_episodes: tuple[int, ...] = ()
    """The episodes the order reads the file onto, when the verdict is `OTHER_ORDER_OUTSIDE_WINDOW`."""


@dataclass(frozen=True, slots=True)
class BatchPlacements:
    """The outcome of placing the files in a torrent's folder on resolved episode ids, one verdict per file."""

    placements: tuple[Placement, ...]
    """One per name in `PlacementBatch.new_names`, batch order."""

    @property
    def placed(self) -> dict[str, tuple[int, ...]]:
        """Normalized file name -> episode ids for every placed file (each id in the window, used once)."""

        return {p.name: p.ids for p in self.placements if p.verdict.placed}

    @property
    def unplaced(self) -> tuple[str, ...]:
        """The files nothing placed and nothing excluded: the caller warns, never guesses."""

        return tuple(p.name for p in self.placements if not p.verdict.placed and not p.verdict.excluded)

    @property
    def excluded(self) -> tuple[Placement, ...]:
        """The files not this pending import's under its current claims, and how (`PlacementVerdict.excluded`)."""

        return tuple(p for p in self.placements if p.verdict.excluded)

    @property
    def hand_import_files(self) -> tuple[str, ...]:
        """The unplaced files left for a hand import (`PlacementVerdict.left_for_hand_import`)."""

        return tuple(p.name for p in self.placements if p.verdict.left_for_hand_import)

    @property
    def order_only(self) -> dict[str, OrderOnlyKey]:
        """Normalized file name -> the order and key that read it, for every `ORDER_ONLY` file."""

        return {p.name: p.order_only for p in self.placements if p.order_only is not None}


NO_FILE_MAP: FileEpisodeMap = MappingProxyType({})
"""No file map."""


@dataclass(frozen=True, slots=True)
class PlacementBatch:
    """A torrent's files to place, with the parses of the WHOLE torrent and its file map.

    `parsed` may cover MORE files than `to_place`. Every parsed file is read as evidence, placed or not, and a name
    `file_map` holds is never placed, in `to_place` or not.
    """

    to_place: Sequence[str]
    """Normalized file names in SeaDex order (the order only fixes deterministic output), kept as a tuple."""

    parsed: Mapping[str, ParsedFileInfo | None]
    """Series-agnostic parse per file (None when Sonarr's parse was unavailable)."""

    file_map: FileEpisodeMap = NO_FILE_MAP
    """The pending import's file map, normalized name -> episode ids, the ids kept as tuples. Its ids count as used
    wherever the target admits them, and the misnumbered specials pass checks each member in the file map sits on its
    aliased special."""

    new_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    """The distinct names to place, batch order: `to_place` less the names in the file map."""

    all_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    """The distinct names the placer reads: the names to place, then the rest of `parsed` (file map, gone). A name
    in the file map with no parse entry is left out."""

    all_parses_read: bool = field(init=False, repr=False, compare=False)
    """Every parse came from Sonarr this run: no miss (None). File map and gone files are parsed by name too, so a
    miss on any of them counts, as does a file to place that has no parse entry."""

    def __post_init__(self) -> None:
        # Detach from the caller's collections, then wrap read-only. The ids are tupled whatever the type says: a
        # cast or an untyped script can hand lists, and a list never equals `(ep_id,)` in `_aliased_specials`, so a
        # file map file on its aliased special would read as off it and the misnumbered batch would lose its aliases.
        object.__setattr__(self, "to_place", tuple(self.to_place))
        object.__setattr__(self, "parsed", MappingProxyType(dict(self.parsed)))
        object.__setattr__(
            self, "file_map", MappingProxyType({name: tuple(ids) for name, ids in self.file_map.items()})
        )
        names = tuple(name for name in dict.fromkeys(self.to_place) if name not in self.file_map)
        torrent_names = tuple(dict.fromkeys([*names, *self.parsed]))
        object.__setattr__(self, "new_names", names)
        object.__setattr__(self, "all_names", torrent_names)
        object.__setattr__(self, "all_parses_read", all(self.parsed.get(name) is not None for name in torrent_names))


type EpisodeBySize = Mapping[int, int]
"""File size -> episode id, from SeaDex releases with a single video file listed under an entry that covers one
episode. A file with that exact size is that episode."""


@dataclass(frozen=True, slots=True)
class TorrentListing:
    """The windows of the titles SeaDex lists one torrent under, and those titles' special aliases."""

    windows: tuple[TitleWindow, ...] = ()
    """One window per title in the listing, in AniList id order."""

    special_aliases: SpecialAliases = NO_ALIASES
    """The union of those titles' special aliases, minus any TMDB number two titles pair differently. Empty when
    the titles pair against different TMDB shows."""

    ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """Union of the titles' episode ids."""

    def __post_init__(self) -> None:
        # SeaDex can list one hash twice under an entry, so the first window per AniList id wins.
        kept: dict[int, TitleWindow] = {}
        for window in self.windows:
            kept.setdefault(window.al_id, window)
        entries = tuple(sorted(kept.values(), key=lambda window: window.al_id))
        object.__setattr__(self, "windows", entries)
        # Copied before it's wrapped, so a caller's later edits can't reach the listing.
        object.__setattr__(self, "special_aliases", MappingProxyType(dict(self.special_aliases)))
        object.__setattr__(self, "ids", frozenset(ep_id for window in entries for ep_id in window.ordered_episode_ids))


EMPTY_LISTING = TorrentListing()
"""A torrent no entry lists. Listings match by infohash, so a release without one always gets this."""

type SizeMatches = Mapping[str, int]
"""Normalized file name -> the episode id its size matched."""

NO_SIZE_MATCHES: SizeMatches = MappingProxyType({})
"""No files placed by size: no size matched, or the episodes by size weren't read."""


@dataclass(frozen=True, slots=True)
class SeadexEvidence:
    """What SeaDex's listings for the series tell the placer about one torrent, as this run read them."""

    listing: TorrentListing = EMPTY_LISTING
    """The torrent's listing and its titles' special aliases: the ids a numbered batch of specials is checked
    against, and the aliases that can place a misnumbered one. Empty when no entry lists the torrent, which turns the
    misnumbered specials check off."""

    size_matches: SizeMatches = NO_SIZE_MATCHES
    """File name -> episode id, for the files whose size matched an episode (see `EpisodeBySize`)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "size_matches", MappingProxyType(dict(self.size_matches)))

    @property
    def ids(self) -> frozenset[int]:
        """Every episode id of the titles SeaDex lists this torrent under, this title included."""

        return self.listing.ids

    @property
    def special_aliases(self) -> SpecialAliases:
        """The TMDB -> TVDB specials numbers of the listing's titles (`TorrentListing.special_aliases`)."""

        return self.listing.special_aliases


NO_EVIDENCE = SeadexEvidence()
"""Nothing from the listings: no ids and no size matches, so the size placement and the misnumbered specials check
do nothing."""


@dataclass(frozen=True, slots=True)
class PlacementTarget:
    """The window's episodes a placement batch may place onto, plus what the placer needs to place onto them.

    `episode_ids` is the FULL id list and `used` the ids already used: an earlier window's, plus the file map's the
    placer adds (`PlacerState.target`). The two stay separate so a pending import whose file map covers every id
    still keeps to its window, while an EMPTY `episode_ids` means no window at all (`no_window`): the exact pass
    then places a name's parsed season and episode against the whole episode index instead of placing nothing
    forever.
    """

    episode_ids: Sequence[int]
    """The title's resolved episode ids, season order, the file map's included, kept as a tuple. A zero id still
    makes the window real. It is never placed."""

    index: EpisodeIndex
    """ALL the series' episodes: the `(season, episode)` map every key resolves
    through (membership in `episode_ids` limits it to the window, so an exact parse
    outside our title is rejected), plus the titles and absolute numbers the
    episode title matches read."""

    used: frozenset[int] = frozenset()
    """Ids already used, by an earlier window or the file map. Never placed again."""

    title_names: TitleNames = field(default_factory=TitleNames)
    """The series and AniList titles. Episode titles and extras words are read against them, and they break
    ties when several sequences or files with no parsed number could be placed on the unused ids."""

    seadex: SeadexEvidence = NO_EVIDENCE
    """What SeaDex's listings tell us about this torrent this run."""

    earlier: FileEpisodeMap = field(default_factory=dict[str, tuple[int, ...]])
    """Normalized name -> the ids an earlier window placed it on, whether or not this window admits them (`after`).
    The other-orders pass reads them as already placed."""

    episode_count: int | None = None
    """AniList's episode count for the title, None when the window carries none (`TitleWindow.episode_count`)."""

    real_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """The resolved ids that can be placed. A zero id keeps the window real but is never one."""

    def __post_init__(self) -> None:
        # Detached from the caller's list: `real_ids` is derived from the list once.
        object.__setattr__(self, "episode_ids", tuple(self.episode_ids))
        object.__setattr__(self, "real_ids", frozenset(i for i in self.episode_ids if i))
        object.__setattr__(self, "earlier", MappingProxyType(dict(self.earlier)))

    @property
    def id_by_key(self) -> Mapping[EpisodeKey, int]:
        """`(season, episode)` -> id over the whole series."""

        return self.index.id_by_key

    @property
    def no_window(self) -> bool:
        """No window at all: no resolved ids (never just every id in the file map)."""

        return not self.episode_ids

    def admits(self, ep_id: int) -> bool:
        """Whether a file may be placed on the episode: one of the resolved set, or any with no window."""

        return self.no_window or ep_id in self.real_ids

    def using(self, ids: Iterable[int]) -> Self:
        """The target with `ids` used too: the ones it admits (every one with no window)."""

        return replace(self, used=self.used | frozenset(i for i in ids if self.admits(i)))

    def after(self, earlier: FileEpisodeMap) -> Self:
        """The target of a window placed after others: `earlier` are their placements, the ids used where admitted."""

        return replace(self.using(ep_id for ids in earlier.values() for ep_id in ids), earlier=earlier)

    @property
    def specials_window(self) -> bool:
        """A window of specials only, the one shape a listing judges."""

        return not self.no_window and all_specials(self.index, self.real_ids)

    def specials_listing(self) -> frozenset[int] | None:
        """The listing a numbered batch of specials is judged by: read, all specials, covering the window. Else None."""

        listed = self.seadex.ids
        if not (self.specials_window and listed and self.real_ids <= listed):
            return None
        return listed if all_specials(self.index, listed) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class GrabbingTitle:
    """What one title's files are placed against: its indexes, its names, and what AniList and Sonarr say of it."""

    al_id: int
    """The title claiming the placement: a stored claim of its own is replaced by the fresh window."""

    series_id: int
    """The Sonarr series the title's episodes belong to."""

    title_index: EpisodeIndex
    """The title's episodes: the resolved set, and what the window and preowned lookups cover."""

    series_index: EpisodeIndex
    """The WHOLE series (empty when the series list could not be read, which places nothing)."""

    title_names: TitleNames
    """The series and AniList titles the placement breaks ties by, stored on every pending import update."""

    episode_count: int | None
    """AniList's episode count for the title (`SonarrEpisodes.episode_count`), None when AniList gives none."""

    alternate_titles: tuple[str, ...]
    """The series' other titles in Sonarr (`SonarrSeries.alternateTitles`), in Sonarr's order."""

    aired_order: bool
    """Whether Sonarr numbers the series by TVDB's aired order (`SonarrEpisodes.aired_order`)."""

    window: TitleWindow = field(init=False, repr=False, compare=False)
    """The title's own window, built as a listing builds each title's (`TitleWindow.of`)."""

    def __post_init__(self) -> None:
        window = TitleWindow.of(
            al_id=self.al_id,
            series_id=self.series_id,
            ep_list=self.title_index.by_id.values(),
            names=self.title_names,
            episode_count=self.episode_count,
        )
        object.__setattr__(self, "window", window)

    @property
    def can_place(self) -> bool:
        """Whether there is a title index and a fetched series index to place against.

        An empty title index must not pass as "no window" (which places against the whole series index), and an
        unread series index places nothing: a grab-time verdict is final.
        """

        return bool(self.title_index.by_id and self.series_index.id_by_key)


@dataclass(frozen=True, slots=True)
class FetchedListings:
    """What this run read from one series' SeaDex entries: listings by infohash, plus the episodes by size."""

    by_hash: Mapping[str, TorrentListing | None]
    """Infohash -> listing, or None if an entry listing that torrent couldn't be read."""

    episode_by_size: EpisodeBySize | None
    """The series' episodes by size, minus any size that two entries give to different episodes. None if any
    entry's episodes couldn't be read, since that entry's sizes would be missing."""

    def __post_init__(self) -> None:
        # Detach from the caller's dicts, then wrap read-only.
        object.__setattr__(self, "by_hash", MappingProxyType(dict(self.by_hash)))
        if self.episode_by_size is not None:
            object.__setattr__(self, "episode_by_size", MappingProxyType(dict(self.episode_by_size)))

    def listing(self, infohash: str) -> TorrentListing | None:
        """The torrent's listing: None if it couldn't be read, `EMPTY_LISTING` if no entry lists it."""

        return self.by_hash.get(infohash, EMPTY_LISTING)
