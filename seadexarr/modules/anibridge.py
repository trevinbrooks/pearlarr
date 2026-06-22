"""Parsing and indexed lookup for anibridge-mappings (v3+).

The anibridge dataset (https://github.com/anibridge/anibridge-mappings) is a
*directional* graph: a JSON object whose keys and values are descriptors of the
form "provider: id[:scope]" (e.g. "anilist:269", ``tvdb_show:74796: s2``).
Every value is a map "{target_descriptor: {source_range: target_range}}".

Every AniList id present in the dataset also appears as its own source key, so
this module parses the "anilist:*" entries once into a per-AniList record plus
a set of reverse indexes (tvdb/tmdb/imdb -> AniList), giving O(1) lookups instead
of the linear scans the old per-id mapping format required.

Episode ranges are kept in TVDB/TMDB numbering: for "anilist:269" ->
"tvdb_show:74796:s2" with value "{"21-41": "1-21"}" the *target* side
(``1-21") is the season-2 TVDB episode range, which is exactly what episode
filtering in Sonarr needs.
"""

import logging
from collections import defaultdict
from typing import Any


def _parse_descriptor(descriptor: str) -> tuple[str, str | None, str | None]:
    """Split a "provider:id[:scope]" descriptor into its parts.

    Args:
        descriptor (str): e.g. "tvdb_show:74796:s2" or "anilist:269"

    Returns:
        tuple[str, str | None, str | None]: (provider, id, scope)
    """

    parts = descriptor.split(":")
    provider = parts[0]
    pid = parts[1] if len(parts) > 1 else None
    scope = parts[2] if len(parts) > 2 else None
    return provider, pid, scope


def _parse_season(scope: str | None) -> int | None:
    """Parse a show scope like "s2" into an integer season number.

    Args:
        scope (str | None): Scope portion of a descriptor

    Returns:
        int | None: Season number, or None if the scope isn't "s<digits>"
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
        target (str): e.g. "1-21", "1-6,8-13", "14-|2", "1"

    Returns:
        list[tuple[int, int | None]]: (start, end); the end is None for open-ended
    """

    ranges = []
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


def _first(values: list) -> Any:
    """Return the first value of a sequence, or None when empty.

    Args:
        values (list): Collected ids for a provider
    """

    return values[0] if values else None


class AniBridge:
    """Indexed view over an anibridge-mappings graph.

    Parses the graph once at construction and exposes id-keyed lookups plus the
    sets of ids needed to filter a Sonarr/Radarr library down to anime.

    Args:
        graph (dict): Raw anibridge mappings JSON (descriptor -> targets)
        logger (logging.Logger | None): Optional logger for skipped descriptors
    """

    def __init__(self, graph: dict, logger: logging.Logger | None = None) -> None:

        self.logger = logger

        # AniList id (int) -> record dict of the ids/episode-maps it points to
        self.by_anilist = {}

        # Reverse indexes: external id -> set of AniList ids
        self.tvdb_index = defaultdict(set)
        self.tmdb_show_index = defaultdict(set)
        self.tmdb_movie_index = defaultdict(set)
        self.imdb_index = defaultdict(set)

        self._parse(graph or {})

        # Precomputed id sets for cheap library filtering
        self.all_tvdb_ids = set(self.tvdb_index)
        self.all_tmdb_movie_ids = set(self.tmdb_movie_index)
        self.all_imdb_ids = set(self.imdb_index)

    def __bool__(self) -> bool:
        return bool(self.by_anilist)

    def __len__(self) -> int:
        return len(self.by_anilist)

    def _parse(self, graph: dict) -> None:
        """Build per-AniList records and reverse indexes from the graph.

        Args:
            graph (dict): Raw anibridge mappings JSON
        """

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

            record = {
                "anidb_id": None,
                "mal_ids": [],
                "tvdb_shows": {},  # tvdb_id -> {season -> [(start, end|None)]}
                "tmdb_shows": {},  # tmdb_id -> {season -> [(start, end|None)]}
                "tmdb_movie_ids": [],
                "tvdb_movie_ids": [],
                "imdb_ids": [],
            }

            for target, ep_map in targets.items():
                self._add_target(record, anilist_id, target, ep_map)

            self.by_anilist[anilist_id] = record

    def _add_target(self, record: dict, anilist_id: int, target: str, ep_map: dict) -> None:
        """Fold a single target descriptor into an AniList record.

        Args:
            record (dict): The AniList record being built
            anilist_id (int): AniList id owning this record
            target (str): Target descriptor (e.g. "tvdb_show:74796:s2")
            ep_map (dict): {source_range: target_range} for this target
        """

        provider, pid, scope = _parse_descriptor(target)

        if provider == "anidb":
            if record["anidb_id"] is None:
                anidb_id = self._as_int(pid)
                if anidb_id is not None:
                    record["anidb_id"] = anidb_id

        elif provider == "mal":
            mal_id = self._as_int(pid)
            if mal_id is not None:
                record["mal_ids"].append(mal_id)

        elif provider in ("tvdb_show", "tmdb_show"):
            ext_id = self._as_int(pid)
            if ext_id is None:
                return
            shows = record["tvdb_shows"] if provider == "tvdb_show" else record["tmdb_shows"]
            index = self.tvdb_index if provider == "tvdb_show" else self.tmdb_show_index

            seasons = shows.setdefault(ext_id, {})
            index[ext_id].add(anilist_id)

            season = _parse_season(scope)
            if season is None:
                # Keep the id discoverable even if the season scope is malformed;
                # an empty season map simply selects no episodes for it.
                self._debug("anibridge: unparseable show scope %r for anilist:%s", target, anilist_id)
                return

            ranges = seasons.setdefault(season, [])
            for tgt in (ep_map or {}).values():
                ranges.extend(_parse_ranges(tgt))

        elif provider == "tmdb_movie":
            movie_id = self._as_int(pid)
            if movie_id is not None:
                record["tmdb_movie_ids"].append(movie_id)
                self.tmdb_movie_index[movie_id].add(anilist_id)

        elif provider == "tvdb_movie":
            movie_id = self._as_int(pid)
            if movie_id is not None:
                record["tvdb_movie_ids"].append(movie_id)

        elif provider in ("imdb_movie", "imdb_show"):
            if pid:
                record["imdb_ids"].append(pid)
                self.imdb_index[pid].add(anilist_id)

    def _as_int(self, value: str | None) -> int | None:
        """Coerce a descriptor id to int, returning None on failure.

        Args:
            value (str | None): Raw id from a descriptor
        """

        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _debug(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.debug(msg, *args)

    def _consumer_entry(
        self,
        anilist_id: int,
        tvdb_id: int | None = None,
        tmdb_show_id: int | None = None,
    ) -> dict:
        """Build the mapping dict consumed by the Sonarr/Radarr pipeline.

        The entry mirrors the field names the rest of the code already reads.
        "tvdb_mappings" (season -> ranges) is only attached when the lookup is
        scoped to a tvdb id that has season data, so it doubles as the
        "this is an anibridge series" marker used by "get_ep_list".

        Args:
            anilist_id (int): AniList id
            tvdb_id (int | None): tvdb id this entry is scoped to, if any
            tmdb_show_id (int | None): tmdb show id this entry is scoped to, if any

        Returns:
            dict: Consumer-facing mapping entry
        """

        record = self.by_anilist[anilist_id]

        entry = {
            "tvdb_id": tvdb_id if tvdb_id is not None else next(iter(record["tvdb_shows"]), None),
            "anidb_id": record["anidb_id"],
            "imdb_id": _first(record["imdb_ids"]),
            "tmdb_show_id": tmdb_show_id if tmdb_show_id is not None else next(iter(record["tmdb_shows"]), None),
            "tmdb_movie_id": _first(record["tmdb_movie_ids"]),
            "mal_id": _first(record["mal_ids"]),
            "source": "anibridge",
        }

        if tvdb_id is not None and tvdb_id in record["tvdb_shows"]:
            entry["tvdb_mappings"] = record["tvdb_shows"][tvdb_id]

        return entry

    def lookup_by_tvdb(self, tvdb_id: int) -> dict:
        """Return "{anilist_id: entry}" for AniList ids mapped to a tvdb id.

        Args:
            tvdb_id (int): TVDB series id
        """

        return {
            anilist_id: self._consumer_entry(anilist_id, tvdb_id=tvdb_id)
            for anilist_id in self.tvdb_index.get(tvdb_id, ())
        }

    def lookup_by_tmdb(self, tmdb_id: int, tmdb_type: str = "movie") -> dict:
        """Return "{anilist_id: entry}" for AniList ids mapped to a tmdb id.

        Args:
            tmdb_id (int): TMDB id
            tmdb_type (str): "movie" or "show"
        """

        if tmdb_type == "show":
            return {
                anilist_id: self._consumer_entry(anilist_id, tmdb_show_id=tmdb_id)
                for anilist_id in self.tmdb_show_index.get(tmdb_id, ())
            }

        return {
            anilist_id: self._consumer_entry(anilist_id)
            for anilist_id in self.tmdb_movie_index.get(tmdb_id, ())
        }

    def lookup_by_imdb(self, imdb_id: str) -> dict:
        """Return "{anilist_id: entry}" for AniList ids mapped to an IMDb id.

        Args:
            imdb_id (str): IMDb id (e.g. "tt0094625")
        """

        return {
            anilist_id: self._consumer_entry(anilist_id)
            for anilist_id in self.imdb_index.get(imdb_id, ())
        }
