"""Parsing and indexed lookup for anibridge-mappings (v3+).

The anibridge dataset (https://github.com/anibridge/anibridge-mappings) is a
*directional* graph: a JSON object whose keys and values are descriptors of the
form "provider: id[:scope]" (e.g. "anilist:269", `tvdb_show:74796: s2`).
Every value is a map "{target_descriptor: {source_range: target_range}}".

Every AniList id present in the dataset also appears as its own source key, so
this module parses the "anilist:*" entries once into a per-AniList record plus
a set of reverse indexes (tvdb/tmdb/imdb -> AniList), giving O(1) lookups.

Episode ranges are kept in TVDB/TMDB numbering: for "anilist:269" ->
"tvdb_show:74796:s2" with value "{"21-41": "1-21"}" the *target* side
(`1-21`) is the season-2 TVDB episode range, which is exactly what episode
filtering in Sonarr needs.

When a record maps its specials onto one TVDB show and one TMDB show,
`_special_aliasing` pairs the two numberings by position within each source
range, so a specials pack numbered in TMDB's order can be placed on the TVDB
specials. `AniBridge._unshared_aliasing` then drops any alias onto a TVDB
special another entry of the show also maps.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, NamedTuple

from .mapping_store import (
    AniBridgeAliasHit,
    AniBridgeAliasRow,
    AniBridgeEntryRow,
    AniBridgeRangeRow,
    AniBridgeRows,
    AniBridgeXrefRow,
    AnimeIdColumn,
    MappingStore,
)
from .seadex_types import SpecialAliasing, TvdbMappings, coerce_int, season_holds, unambiguous

type AniBridgeGraph = dict[str, dict[str, dict[str, str]]]
"""Raw anibridge-mappings JSON: descriptor -> {target_descriptor -> {src: tgt}}."""

type AniBridgeEntry = dict[str, Any]
"""One consumer-facing mapping entry (mixed-typed). `mappings` reads it via
`_entry_from_raw`. Stays a loose `dict` - it is the raw->typed boundary,
not the typed domain."""

type AniBridgeLookup = dict[int, AniBridgeEntry]
"""A `lookup_by_*` result: AniList id -> its consumer entry."""


class TvdbShowMaps(NamedTuple):
    """The per-show fields a TVDB lookup adds to an entry, named as the entry dict keys them."""

    tvdb_mappings: TvdbMappings
    special_aliasing: SpecialAliasing | None
    """None when the record's specials don't pair for this show, and then the entry leaves the key out."""


@dataclass
class AniBridgeRecord:
    """The per-AniList record built incrementally while parsing the graph.

    One record accumulates every external id and season->episode-range map an
    `anilist:*` entry points to. It is *mutable and built up* by
    `AniBridge._add_target` (appending to the list/dict fields as targets
    are folded in), so the collection fields default-construct empty rather than
    being passed at once. `_consumer_entry` then reads attributes off it.
    """

    anidb_id: int | None = None
    tvdb_shows: dict[int, TvdbMappings] = field(default_factory=dict[int, TvdbMappings])
    """Keyed by external id and holds a `TvdbMappings` (season -> inclusive
    `(start, end)` ranges) per id."""
    tmdb_movie_ids: list[int] = field(default_factory=list[int])
    imdb_ids: list[str] = field(default_factory=list[str])
    special_aliasing: dict[int, SpecialAliasing] = field(default_factory=dict[int, SpecialAliasing])
    """TVDB show id -> the record's specials aliasing for that show. `AniBridge._parse` fills it per record, then
    runs it through `_unshared_aliasing` once every record exists."""


def _parse_descriptor(descriptor: str) -> tuple[str, str | None, str | None]:
    """Split a "provider:id[:scope]" descriptor into its parts.

    Args:
        descriptor: e.g. "tvdb_show:74796:s2" or "anilist:269"

    Returns:
        The `(provider, id, scope)` parts. The id and scope are None when absent.
    """

    parts = descriptor.split(":")
    provider = parts[0]
    pid = parts[1] if len(parts) > 1 else None
    scope = parts[2] if len(parts) > 2 else None
    return provider, pid, scope


def _parse_season(scope: str | None) -> int | None:
    """Parse a show scope like "s2" into an integer season number.

    Returns:
        The season number, or None if the scope isn't "s<digits>"
    """

    if not scope or not scope.startswith("s"):
        return None
    try:
        return int(scope[1:])
    except ValueError:
        return None


def _parse_ranges(target: str) -> list[tuple[int, int | None]]:
    """Parse a target range string into a list of inclusive (start, end) tuples.

    Handles comma-separated non-contiguous segments and open-ended ranges and
    drops any "|ratio" suffix (a ratio describes mapping density, not which
    episodes are covered).

    Args:
        target: e.g. "1-21", "1-6,8-13", "14-|2", "1"

    Returns:
        The `(start, end)` pairs. The end is None for an open-ended range
    """

    pieces = (_parse_piece(piece) for piece in str(target).split("|")[0].split(","))
    return [piece for piece in pieces if piece is not None]


def _parse_piece(piece: str) -> tuple[int, int | None] | None:
    """One range piece as inclusive `(start, end)`: `n`, `a-b`, or `a-` with an end of None. None for anything else."""

    start_text, dash, end_text = piece.strip().partition("-")
    try:
        start = int(start_text)
        if not dash:
            end = start
        elif end_text:
            end = int(end_text)
        else:
            end = None
    except ValueError:
        return None
    return start, end


def _plain_ranges(text: str) -> list[range] | None:
    """Each comma piece of `text` as a `range`, or None unless every piece is `n` or `a-b` with a <= b.

    So a ratio suffix (`3-4|2`), an open end (`5-`), or a junk piece anywhere gives None.
    """

    if "|" in text:
        return None
    ranges: list[range] = []
    for piece in text.split(","):
        parsed = _parse_piece(piece)
        if parsed is None:
            return None
        first, last = parsed
        if last is None or last < first:
            return None
        ranges.append(range(first, last + 1))
    return ranges


class SpecialsTarget(NamedTuple):
    """One season-0 target of a record: its show id (None if not a number) and its source -> target range map."""

    show_id: int | None
    ep_map: Mapping[str, str]


def _specials_maps(targets: Mapping[str, Mapping[str, str]], provider: str) -> list[SpecialsTarget]:
    """The record's season-0 targets on `provider` (`tvdb_show` or `tmdb_show`)."""

    found: list[SpecialsTarget] = []
    for target, ep_map in targets.items():
        kind, pid, scope = _parse_descriptor(target)
        if kind == provider and _parse_season(scope) == 0:
            found.append(SpecialsTarget(coerce_int(pid), ep_map or {}))
    return found


def _special_aliasing(targets: Mapping[str, Mapping[str, str]]) -> dict[int, SpecialAliasing]:
    """A record's TMDB specials numbers paired with its TVDB ones, keyed by the TVDB show. Empty if nothing pairs.

    Needs exactly one TVDB and one TMDB specials target, each with a show id. In each source range both map,
    the numbers pair by position when the source and both targets are plain ranges of the same width
    (`_plain_ranges`). A TMDB number that pairs two ways is dropped.
    """

    tvdb, tmdb = _specials_maps(targets, "tvdb_show"), _specials_maps(targets, "tmdb_show")
    if len(tvdb) != 1 or len(tmdb) != 1:
        return {}
    (tvdb_id, tvdb_map), (tmdb_id, tmdb_map) = tvdb[0], tmdb[0]
    if tvdb_id is None or tmdb_id is None:
        return {}
    pairs: list[tuple[int, int]] = []
    for source, tvdb_range in tvdb_map.items():
        if (tmdb_range := tmdb_map.get(source)) is None:
            continue
        source_ranges, tvdb_ranges, tmdb_ranges = (_plain_ranges(text) for text in (source, tvdb_range, tmdb_range))
        if source_ranges is None or tvdb_ranges is None or tmdb_ranges is None:
            continue
        if len({sum(map(len, ranges)) for ranges in (source_ranges, tvdb_ranges, tmdb_ranges)}) != 1:
            continue
        pairs.extend(zip(chain.from_iterable(tmdb_ranges), chain.from_iterable(tvdb_ranges), strict=True))
    aliases = unambiguous(pairs)
    return {tvdb_id: SpecialAliasing(tmdb_id, aliases)} if aliases else {}


def _aliasing_by_anilist(hits: Iterable[AniBridgeAliasHit]) -> dict[int, SpecialAliasing]:
    """The stored alias rows of one TVDB show regrouped per AniList id, in row order."""

    found: dict[int, tuple[int, dict[int, int]]] = {}
    for hit in hits:
        _, aliases = found.setdefault(hit.anilist_id, (hit.tmdb_id, {}))
        aliases[hit.tmdb_number] = hit.tvdb_number
    return {anilist_id: SpecialAliasing(tmdb_id, aliases) for anilist_id, (tmdb_id, aliases) in found.items()}


def _first[T](values: list[T]) -> T | None:
    """Return the first value of a sequence, or None when empty."""

    return values[0] if values else None


class AniBridge:
    """Indexed view over an anibridge-mappings graph (in-memory or SQL-backed).

    Two interchangeable backings behind one interface (`lookup_by_*` / the
    `all_*` id sets / `__len__` / `__bool__`):

    * **Graph-backed** (`AniBridge(graph)`): parses the graph once into per-AniList
      records + reverse indexes. This is the parser/populator (its `to_rows`
      feeds the SQL store) and the test oracle.
    * **SQL-backed** (`from_store`): answers the same lookups from
      `mappings.db` without holding the parsed graph in memory - the runtime path.

    Args:
        graph: Raw anibridge mappings JSON (descriptor -> targets)
        logger: Optional logger for skipped descriptors
    """

    def __init__(self, graph: AniBridgeGraph, logger: logging.Logger | None = None) -> None:

        self.logger = logger

        # SQL backing, set only by `from_store`. None means graph-backed (below).
        self._store: MappingStore | None = None

        # AniList id (int) -> the record of the ids/episode-maps it points to
        self.by_anilist: dict[int, AniBridgeRecord] = {}

        # Reverse indexes: external id -> set of AniList ids
        self.tvdb_index: dict[int, set[int]] = defaultdict(set)
        self.tmdb_movie_index: dict[int, set[int]] = defaultdict(set)
        self.imdb_index: dict[str, set[int]] = defaultdict(set)

        self._parse(graph or {})

        # Precomputed id sets for cheap library filtering
        self.all_tvdb_ids: set[int] = set(self.tvdb_index)
        self.all_tmdb_movie_ids: set[int] = set(self.tmdb_movie_index)
        self.all_imdb_ids: set[str] = set(self.imdb_index)

        # Entry count, cached once: by_anilist is fixed after _parse, so __len__ /
        # __bool__ never recompute it.
        self._len = len(self.by_anilist)

    @classmethod
    def from_store(cls, store: MappingStore) -> "AniBridge":
        """Build a SQL-backed view that answers lookups from `mappings.db`.

        Holds only the store handle plus the (small) `all_*` id sets loaded once.
        The per-AniList records and reverse indexes live in SQL, so the ~25MB parsed
        graph is never resident. `lookup_by_*` query the store on demand.

        Args:
            store: Store whose anibridge tables are already populated.
        """

        self = cls.__new__(cls)
        self.logger = None
        self._store = store
        # The graph-backed fields stay empty. The store answers everything.
        self.by_anilist = {}
        self.tvdb_index = defaultdict(set)
        self.tmdb_movie_index = defaultdict(set)
        self.imdb_index = defaultdict(set)
        # anibridge_distinct's per-axis overloads type each set (tvdb/tmdb ints,
        # imdb strs). The store raises on a mismatched stored type.
        self.all_tvdb_ids = store.anibridge_distinct("tvdb")
        self.all_tmdb_movie_ids = store.anibridge_distinct("tmdb_movie")
        self.all_imdb_ids = store.anibridge_distinct("imdb")
        # Entry count, fetched once: the store is immutable for this view's lifetime
        # (populated before from_store, read-only after), so a per-call COUNT(*) - an
        # O(rows) scan that get_anilist_ids would trigger twice per item - is wasteful.
        self._len = store.anibridge_len()
        return self

    def to_rows(self) -> AniBridgeRows:
        """Flatten this (graph-backed) view into store row tuples.

        Persists the *computed* consumer-entry picks (`_first`) and the already
        filtered specials aliases, so the SQL backing reproduces `_consumer_entry`
        with zero re-derivation.

        Returns:
            The `AniBridgeRows` row lists for `MappingStore.replace_anibridge`.
        """

        entries: list[AniBridgeEntryRow] = []
        ranges: list[AniBridgeRangeRow] = []
        alias_rows: list[AniBridgeAliasRow] = []
        for anilist_id, record in self.by_anilist.items():
            entries.append(
                AniBridgeEntryRow(
                    anilist_id=anilist_id,
                    anidb_id=record.anidb_id,
                    imdb_id=_first(record.imdb_ids),
                    tmdb_movie_id=_first(record.tmdb_movie_ids),
                ),
            )
            for tvdb_id, seasons in record.tvdb_shows.items():
                for season, range_list in seasons.items():
                    if not range_list:
                        # Present-but-empty season: see the NULL start_ep marker
                        # note on anibridge_tvdb_range in mapping_store.py.
                        ranges.append(AniBridgeRangeRow(anilist_id, tvdb_id, season, None, None))
                        continue
                    for start, end in range_list:
                        ranges.append(AniBridgeRangeRow(anilist_id, tvdb_id, season, start, end))
            for tvdb_id, (tmdb_id, aliases) in record.special_aliasing.items():
                alias_rows.extend(
                    AniBridgeAliasRow(anilist_id, tvdb_id, tmdb_id, tmdb_number, tvdb_number)
                    for tmdb_number, tvdb_number in aliases.items()
                )

        xrefs: list[AniBridgeXrefRow] = []
        for axis, index in (
            ("tvdb", self.tvdb_index),
            ("tmdb_movie", self.tmdb_movie_index),
            ("imdb", self.imdb_index),
        ):
            for ext_id, anilist_ids in index.items():
                for anilist_id in anilist_ids:
                    xrefs.append(AniBridgeXrefRow(axis, ext_id, anilist_id))

        return AniBridgeRows(entries, xrefs, ranges, alias_rows)

    def __bool__(self) -> bool:
        return self._len > 0

    def __len__(self) -> int:
        return self._len

    def id_set(self, mapping_key: AnimeIdColumn) -> set[int] | set[str]:
        """The precomputed candidate id set for a Kometa `mapping_key` axis.

        Mirrors `MappingResolver.anime_id_set` so `collect_anime_items` can
        build BOTH sources' candidate-set tuples from one comprehension over the same
        `fields` - instead of a hand-ordered literal that can silently drift out of
        positional alignment with `fields` (the `zip(strict=True)` only checks
        length, not correspondence). The keys are exactly the `mapping_key`s the
        library filter passes (tvdb / tmdb-movie / imdb axes).
        """

        return {
            "tvdb_id": self.all_tvdb_ids,
            "tmdb_movie_id": self.all_tmdb_movie_ids,
            "imdb_id": self.all_imdb_ids,
        }[mapping_key]

    def _parse(self, graph: AniBridgeGraph) -> None:
        """Build per-AniList records and reverse indexes from the graph."""

        for key, targets in graph.items():
            provider, pid, _ = _parse_descriptor(key)
            if provider != "anilist" or pid is None:
                # Reverse links are reconstructed from the anilist-keyed entries,
                # and `$meta` / non-anilist sources are not needed here.
                continue

            try:
                anilist_id = int(pid)
            except ValueError:
                continue

            record = AniBridgeRecord()

            for target, ep_map in targets.items():
                self._add_target(record, anilist_id, target, ep_map)
            record.special_aliasing = _special_aliasing(targets)

            self.by_anilist[anilist_id] = record

        # Filter only now that the loop is done: `_unshared_aliasing` reads every other record of the show.
        for anilist_id, record in self.by_anilist.items():
            if record.special_aliasing:
                record.special_aliasing = self._unshared_aliasing(anilist_id, record.special_aliasing)

    def _unshared_aliasing(
        self, anilist_id: int, aliasing: Mapping[int, SpecialAliasing]
    ) -> dict[int, SpecialAliasing]:
        """The aliasing minus every alias onto a TVDB special another entry of the show maps in its own season 0.

        AniBridge sometimes has a stale specials range on one entry and the right one on another, and an alias
        built from the stale range would put a file on the wrong special. A show left with no alias is dropped.
        """

        unshared: dict[int, SpecialAliasing] = {}
        for tvdb_id, (tmdb_id, aliases) in aliasing.items():
            others = (self.by_anilist[other] for other in self.tvdb_index.get(tvdb_id, ()) if other != anilist_id)
            claimed = [ranges for other in others if (ranges := other.tvdb_shows.get(tvdb_id, {}).get(0)) is not None]
            kept = {
                tmdb_number: tvdb_number
                for tmdb_number, tvdb_number in aliases.items()
                if not any(season_holds(ranges, tvdb_number) for ranges in claimed)
            }
            if kept:
                unshared[tvdb_id] = SpecialAliasing(tmdb_id, kept)
        return unshared

    def _add_target(
        self,
        record: AniBridgeRecord,
        anilist_id: int,
        target: str,
        ep_map: dict[str, str],
    ) -> None:
        """Fold a single target descriptor into an AniList record.

        Args:
            record: The AniList record being built
            anilist_id: AniList id owning this record
            target: Target descriptor (e.g. "tvdb_show:74796:s2")
            ep_map: {source_range: target_range} for this target
        """

        provider, pid, scope = _parse_descriptor(target)

        if provider == "anidb":
            if record.anidb_id is None:
                anidb_id = coerce_int(pid)
                if anidb_id is not None:
                    record.anidb_id = anidb_id

        elif provider == "tvdb_show":
            ext_id = coerce_int(pid)
            if ext_id is None:
                return
            seasons = record.tvdb_shows.setdefault(ext_id, {})
            self.tvdb_index[ext_id].add(anilist_id)

            season = _parse_season(scope)
            if season is None:
                # Keep the id discoverable even if the season scope is malformed.
                # An empty season map simply selects no episodes for it.
                if self.logger is not None:
                    self.logger.debug(f"anibridge: unparseable show scope {target!r} for anilist:{anilist_id}")
                return

            ranges = seasons.setdefault(season, [])
            for tgt in (ep_map or {}).values():
                ranges.extend(_parse_ranges(tgt))

        elif provider == "tmdb_movie":
            movie_id = coerce_int(pid)
            if movie_id is not None:
                record.tmdb_movie_ids.append(movie_id)
                self.tmdb_movie_index[movie_id].add(anilist_id)

        elif provider in ("imdb_movie", "imdb_show") and pid:
            record.imdb_ids.append(pid)
            self.imdb_index[pid].add(anilist_id)

    @staticmethod
    def _entry_dict(
        *,
        anidb_id: int | None,
        imdb_id: str | None,
        tmdb_movie_id: int | None,
        tvdb: TvdbShowMaps | None = None,
    ) -> AniBridgeEntry:
        """The consumer-facing mapping dict, the single shape both backings build.

        "tvdb_mappings" attaches ONLY when `tvdb` is given (never by truthiness): an empty {}
        still attaches, doubling as the "this is an anibridge series" marker. "special_aliasing"
        attaches only when `tvdb` carries one. Stays a loose dict: the raw->typed boundary, not
        the typed domain.
        """

        entry: dict[str, Any] = {
            "anidb_id": anidb_id,
            "imdb_id": imdb_id,
            "tmdb_movie_id": tmdb_movie_id,
            "source": "anibridge",
        }
        if tvdb is not None:
            entry["tvdb_mappings"] = tvdb.tvdb_mappings
            if tvdb.special_aliasing is not None:
                entry["special_aliasing"] = tvdb.special_aliasing
        return entry

    def _consumer_entry(
        self,
        anilist_id: int,
        tvdb_id: int | None = None,
    ) -> AniBridgeEntry:
        """Build the mapping dict consumed by the Sonarr/Radarr pipeline.

        The entry mirrors the field names the rest of the code already reads.
        "tvdb_mappings" (season -> ranges) is only attached when the lookup is
        scoped to a tvdb id that has season data, so it doubles as the "this is an
        anibridge series" marker used by "get_ep_list". "special_aliasing" comes
        along only when the record's specials also pair for that show.
        """

        record = self.by_anilist[anilist_id]
        tvdb = None
        if tvdb_id is not None and tvdb_id in record.tvdb_shows:
            tvdb = TvdbShowMaps(record.tvdb_shows[tvdb_id], record.special_aliasing.get(tvdb_id))
        return self._entry_dict(
            anidb_id=record.anidb_id,
            imdb_id=_first(record.imdb_ids),
            tmdb_movie_id=_first(record.tmdb_movie_ids),
            tvdb=tvdb,
        )

    @staticmethod
    def _ranges_to_mappings(rows: list[tuple[int, int | None, int | None]]) -> TvdbMappings:
        """Rebuild a season -> `[(start, end)]` map from ordered range rows.

        `rows` arrive in populate (insertion) order, so each season's range list
        is rebuilt in the same order the in-memory view appended them. A NULL-start
        marker row creates the season key with an empty list (present-but-empty
        season), exactly mirroring the in-memory `{season: []}`. Season key order
        is irrelevant to dict equality.
        """

        mappings: TvdbMappings = {}
        for season, start, end in rows:
            bucket = mappings.setdefault(season, [])
            if start is not None:
                bucket.append((start, end))
        return mappings

    def _sql_lookup(
        self,
        axis: str,
        ext_id: int | str,
        *,
        tvdb_id: int | None = None,
    ) -> AniBridgeLookup:
        """Batched SQL twin of the graph `lookup_by_*` (on a stored view).

        One xref->entry JOIN fetches every entry mapped to `ext_id` on `axis`.
        For a tvdb-scoped lookup, an xref->range JOIN and an xref->alias JOIN fetch
        all their range and alias rows at once (grouped here by AniList id), so
        resolving k ids costs 3 queries rather than the 1 + 3k point queries a per-id
        approach needs. Reproduces `_consumer_entry` exactly: `tvdb_mappings` attaches
        whenever `tvdb_id` is supplied, and `special_aliasing` whenever the id also has
        alias rows. The only such caller (`lookup_by_tvdb`)
        iterates the tvdb xref, so every resolved id is guaranteed to carry that
        tvdb (matching the in-memory `tvdb_id in record.tvdb_shows` guard).
        """

        store = self._store
        assert store is not None  # only reached on a SQL-backed view

        ranges_by_anilist: dict[int, list[tuple[int, int | None, int | None]]] = {}
        aliasing_by_anilist: dict[int, SpecialAliasing] = {}
        if tvdb_id is not None:
            for hit in store.anibridge_ranges_for(axis, ext_id, tvdb_id):
                ranges_by_anilist.setdefault(hit.anilist_id, []).append((hit.season, hit.start_ep, hit.end_ep))
            aliasing_by_anilist = _aliasing_by_anilist(store.anibridge_aliases_for(axis, ext_id, tvdb_id))

        result: AniBridgeLookup = {}
        for row in store.anibridge_entries_for(axis, ext_id):
            tvdb = None
            if tvdb_id is not None:
                tvdb_mappings = self._ranges_to_mappings(ranges_by_anilist.get(row.anilist_id, []))
                tvdb = TvdbShowMaps(tvdb_mappings, aliasing_by_anilist.get(row.anilist_id))
            result[row.anilist_id] = self._entry_dict(
                anidb_id=row.anidb_id,
                imdb_id=row.imdb_id,
                tmdb_movie_id=row.tmdb_movie_id,
                tvdb=tvdb,
            )
        return result

    def lookup_by_tvdb(self, tvdb_id: int) -> AniBridgeLookup:
        """Return "{anilist_id: entry}" for AniList ids mapped to a TVDB series id."""

        if self._store is not None:
            return self._sql_lookup("tvdb", tvdb_id, tvdb_id=tvdb_id)

        return {
            anilist_id: self._consumer_entry(anilist_id, tvdb_id=tvdb_id)
            for anilist_id in self.tvdb_index.get(tvdb_id, ())
        }

    def lookup_by_tmdb(self, tmdb_id: int) -> AniBridgeLookup:
        """Return "{anilist_id: entry}" for AniList ids mapped to a TMDB movie id."""

        if self._store is not None:
            return self._sql_lookup("tmdb_movie", tmdb_id)

        return {anilist_id: self._consumer_entry(anilist_id) for anilist_id in self.tmdb_movie_index.get(tmdb_id, ())}

    def lookup_by_imdb(self, imdb_id: str) -> AniBridgeLookup:
        """Return "{anilist_id: entry}" for AniList ids mapped to an IMDb id (e.g. "tt0094625")."""

        if self._store is not None:
            return self._sql_lookup("imdb", imdb_id)

        return {anilist_id: self._consumer_entry(anilist_id) for anilist_id in self.imdb_index.get(imdb_id, ())}
