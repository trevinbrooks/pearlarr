"""Parsing and indexed lookup for anibridge-mappings (v3+).

The anibridge dataset (https://github.com/anibridge/anibridge-mappings) is a
*directional* graph: a JSON object whose keys and values are descriptors of the
form "provider: id[:scope]" (e.g. "anilist:269", `tvdb_show:74796: s2`).
Every value is a map "{target_descriptor: {source_range: target_range}}".

Every AniList id present in the dataset also appears as its own source key, so
this module parses the "anilist:*" entries once into a per-AniList record plus
a set of reverse indexes (tvdb/tmdb/imdb -> AniList), giving O(1) lookups instead
of the linear scans the old per-id mapping format required.

Episode ranges are kept in TVDB/TMDB numbering: for "anilist:269" ->
"tvdb_show:74796:s2" with value "{"21-41": "1-21"}" the *target* side
(`1-21`) is the season-2 TVDB episode range, which is exactly what episode
filtering in Sonarr needs.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .mapping_store import (
    AniBridgeEntryRow,
    AniBridgeRangeRow,
    AniBridgeRows,
    AniBridgeXrefRow,
    AnimeIdColumn,
    MappingStore,
)
from .seadex_types import TvdbMappings, coerce_int

type AniBridgeGraph = dict[str, dict[str, dict[str, str]]]
"""Raw anibridge-mappings JSON: descriptor -> {target_descriptor -> {src: tgt}}."""

type AniBridgeEntry = dict[str, Any]
"""One consumer-facing mapping entry (mixed-typed; `mappings` reads it via
`_entry_from_raw`). Stays a loose `dict` - it is the raw->typed boundary,
not the typed domain."""

type AniBridgeLookup = dict[int, AniBridgeEntry]
"""A `lookup_by_*` result: AniList id -> its consumer entry."""


@dataclass
class AniBridgeRecord:
    """The per-AniList record built incrementally while parsing the graph.

    One record accumulates every external id and season->episode-range map an
    `anilist:*` entry points to. It is *mutable and built up* by
    `AniBridge._add_target` (appending to the list/dict fields as targets
    are folded in), so the collection fields default-construct empty rather than
    being passed at once; `_consumer_entry` then reads attributes off it.

    `tvdb_shows` is keyed by external id and holds a `TvdbMappings`
    (season -> inclusive `(start, end)` ranges) per id.
    """

    anidb_id: int | None = None
    tvdb_shows: dict[int, TvdbMappings] = field(default_factory=dict[int, TvdbMappings])
    tmdb_movie_ids: list[int] = field(default_factory=list[int])
    imdb_ids: list[str] = field(default_factory=list[str])


def _parse_descriptor(descriptor: str) -> tuple[str, str | None, str | None]:
    """Split a "provider:id[:scope]" descriptor into its parts.

    Args:
        descriptor: e.g. "tvdb_show:74796:s2" or "anilist:269"

    Returns:
        The `(provider, id, scope)` parts; the id and scope are None when absent.
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
        The `(start, end)` pairs; the end is None for an open-ended range
    """

    ranges: list[tuple[int, int | None]] = []
    target = str(target).split("|")[0]

    for piece in target.split(","):
        piece = piece.strip()
        if not piece:
            continue

        if "-" in piece:
            start_str, _, end_str = piece.partition("-")
            try:
                start = int(start_str)
            except ValueError:
                continue
            end = None
            if end_str:
                try:
                    end = int(end_str)
                except ValueError:
                    continue
        else:
            try:
                start = end = int(piece)
            except ValueError:
                continue

        ranges.append((start, end))

    return ranges


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

        # SQL backing, set only by `from_store`; None means graph-backed (below).
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

        Holds only the store handle plus the (small) `all_*` id sets loaded once;
        the per-AniList records and reverse indexes live in SQL, so the ~25MB parsed
        graph is never resident. `lookup_by_*` query the store on demand.

        Args:
            store: Store whose anibridge tables are already populated.
        """

        self = cls.__new__(cls)
        self.logger = None
        self._store = store
        # The graph-backed fields stay empty; the store answers everything.
        self.by_anilist = {}
        self.tvdb_index = defaultdict(set)
        self.tmdb_movie_index = defaultdict(set)
        self.imdb_index = defaultdict(set)
        # anibridge_distinct's per-axis overloads type each set (tvdb/tmdb ints,
        # imdb strs); the store raises on a mismatched stored type.
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

        Persists the *computed* consumer-entry picks (`_first`) so the SQL
        backing reproduces `_consumer_entry` with zero re-derivation.

        Returns:
            The `AniBridgeRows` row lists for `MappingStore.replace_anibridge`.
        """

        entries: list[AniBridgeEntryRow] = []
        ranges: list[AniBridgeRangeRow] = []
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
                        # Present-but-empty season: a NULL-start marker so it
                        # round-trips as `{season: []}` ("whole season covered")
                        # rather than collapsing to a missing season.
                        ranges.append(AniBridgeRangeRow(anilist_id, tvdb_id, season, None, None))
                        continue
                    for start, end in range_list:
                        ranges.append(AniBridgeRangeRow(anilist_id, tvdb_id, season, start, end))

        xrefs: list[AniBridgeXrefRow] = []
        for axis, index in (
            ("tvdb", self.tvdb_index),
            ("tmdb_movie", self.tmdb_movie_index),
            ("imdb", self.imdb_index),
        ):
            for ext_id, anilist_ids in index.items():
                for anilist_id in anilist_ids:
                    xrefs.append(AniBridgeXrefRow(axis, ext_id, anilist_id))

        return AniBridgeRows(entries, xrefs, ranges)

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

            self.by_anilist[anilist_id] = record

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
                # Keep the id discoverable even if the season scope is malformed;
                # an empty season map simply selects no episodes for it.
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

        elif provider in ("imdb_movie", "imdb_show"):
            if pid:
                record.imdb_ids.append(pid)
                self.imdb_index[pid].add(anilist_id)

    def _consumer_entry(
        self,
        anilist_id: int,
        tvdb_id: int | None = None,
    ) -> AniBridgeEntry:
        """Build the mapping dict consumed by the Sonarr/Radarr pipeline.

        The entry mirrors the field names the rest of the code already reads.
        "tvdb_mappings" (season -> ranges) is only attached when the lookup is
        scoped to a tvdb id that has season data, so it doubles as the
        "this is an anibridge series" marker used by "get_ep_list".
        """

        record = self.by_anilist[anilist_id]

        entry: dict[str, Any] = {
            "anidb_id": record.anidb_id,
            "imdb_id": _first(record.imdb_ids),
            "tmdb_movie_id": _first(record.tmdb_movie_ids),
            "source": "anibridge",
        }

        if tvdb_id is not None and tvdb_id in record.tvdb_shows:
            entry["tvdb_mappings"] = record.tvdb_shows[tvdb_id]

        return entry

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
        ext_id: object,
        *,
        tvdb_id: int | None = None,
    ) -> AniBridgeLookup:
        """Batched SQL twin of the graph `lookup_by_*` (on a stored view).

        One xref->entry JOIN fetches every entry mapped to `ext_id` on `axis`;
        for a tvdb-scoped lookup a second xref->range JOIN fetches all their range
        rows at once (grouped here by AniList id), so resolving k ids costs 2 queries
        rather than the former 1 + 2k point queries. Reproduces `_consumer_entry`
        exactly: `tvdb_mappings` is attached whenever `tvdb_id` is supplied - the
        only such caller (`lookup_by_tvdb`) iterates the tvdb xref, so every
        resolved id is guaranteed to carry that tvdb (matching the in-memory
        `tvdb_id in record.tvdb_shows` guard).
        """

        store = self._store
        assert store is not None  # only reached on a SQL-backed view

        ranges_by_anilist: dict[int, list[tuple[int, int | None, int | None]]] = {}
        if tvdb_id is not None:
            for hit in store.anibridge_ranges_for(axis, ext_id, tvdb_id):
                ranges_by_anilist.setdefault(hit.anilist_id, []).append((hit.season, hit.start_ep, hit.end_ep))

        result: AniBridgeLookup = {}
        for row in store.anibridge_entries_for(axis, ext_id):
            entry: dict[str, Any] = {
                "anidb_id": row.anidb_id,
                "imdb_id": row.imdb_id,
                "tmdb_movie_id": row.tmdb_movie_id,
                "source": "anibridge",
            }
            if tvdb_id is not None:
                entry["tvdb_mappings"] = self._ranges_to_mappings(ranges_by_anilist.get(row.anilist_id, []))
            result[row.anilist_id] = entry
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
