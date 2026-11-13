"""The Sonarr parse cache's rows: one file name's whole `/parse`, its freshness, and the unmatched parse's expiry.

`ParseCacheRows` is bound once on `ArrRunDeps`, so the planner's sweep and the pending
import update builder read one `ParsedFileInfo` per file under one freshness rule.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, NamedTuple, NotRequired, Self, TypedDict, cast

from pydantic import ValidationError

from .arr_models import ParsedFileInfo
from .boundary import Json
from .cache import AbstractCacheStore, stamp_is_fresh
from .json_narrow import is_json_obj
from .stamps import stamp_of

# How long a persisted Sonarr /parse result stays usable before it's re-queried.
# A file name's season and episode are stable, but Sonarr's /parse depends on
# the current library, so a wrong-but-non-empty match could otherwise be trusted
# forever. Re-validate monthly so such a row is parsed again.
SONARR_PARSE_CACHE_TTL_DAYS = 30

# How long an UNMATCHED parse (Sonarr matched no series) stays usable. Pinned to
# the series-id set (see sonarr_episodes.sonarr_series_hash) so adding the
# series makes it stale. This TTL is only a backstop, so it is short.
SONARR_PARSE_UNMATCHED_TTL_DAYS = 7


class ParseFreshness(NamedTuple):
    """The freshness cutoffs for one batch of parses, computed once per call.

    Bundles the values the parse-cache freshness check and writes thread
    together. One is opened per call (the planner's sweep, the pending import
    update build) and passed down to `ParseCacheRows`.
    """

    now_str: str
    """Stamps new rows."""

    matched_cutoff: datetime
    """Bounds a matched row's TTL."""

    unmatched_cutoff: datetime
    """The short unmatched-row backstop."""

    series_hash: str
    """Pins unmatched rows to the current series-id set (so a newly-added series' files are parsed again)."""

    @classmethod
    def open(cls, series_hash: str) -> Self:
        """The cutoffs anchored to one instant (never per file)."""

        now = datetime.now()
        return cls(
            now_str=stamp_of(now),
            matched_cutoff=now - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS),
            unmatched_cutoff=now - timedelta(days=SONARR_PARSE_UNMATCHED_TTL_DAYS),
            series_hash=series_hash,
        )


class ParseCacheRow(TypedDict):
    """One persisted Sonarr `/parse` cache row, keyed by file name."""

    fetched_at: str
    """Stamps the row for TTL eviction."""

    parse: dict[str, Json]
    """The whole `ParsedFileInfo` (see `parse_to_json`): the name's numbers and parsed series name plus the matched
    episodes."""

    series_fp: NotRequired[str]
    """`NotRequired` because only an UNMATCHED row carries it (pinning it to the series-id set), and the
    freshness check dispatches on exactly that presence."""


def parse_to_json(info: ParsedFileInfo) -> dict[str, Json]:
    """The JSON the cache persists for one parse.

    The matched episodes' ids are kept so the placement cross-checks them against the episode index, which never
    resolves a wrong-series match. A series re-add renumbers them, which only weakens the placement until the row
    expires.
    """

    return info.model_dump(mode="json")


def parse_from_json(row: Mapping[str, object]) -> ParsedFileInfo | None:
    """Rebuild a row's parse (the inverse of `parse_to_json`), or None when the row has no readable parse.

    A parse whose values don't validate counts as absent and re-fetches. A missing key validates to its default, so the
    version gate is the field check in `_parse_is_fresh`, which every read passes first.
    """

    raw = row.get("parse")
    if not is_json_obj(raw):
        return None
    try:
        return ParsedFileInfo.model_validate(raw)
    except ValidationError:
        return None


_PARSE_FIELDS = frozenset(ParsedFileInfo.model_fields)
"""The keys `parse_to_json` writes: a row missing one was written before that field existed."""


def _parse_is_fresh(row: Mapping[str, object], *, window: ParseFreshness) -> bool:
    """True if a persisted parse row is still usable.

    A legacy row is stale: one with no whole parse, or whose parse lacks a field of `ParsedFileInfo`, which an
    older build wrote before the field existed. The model would read a missing field as its default, so only this
    check re-parses the row, and the cache warm-up sees it stale too. MATCHED (no `series_fp`): valid for the
    30-day `window.matched_cutoff`. UNMATCHED: valid only while the series-id set is unchanged (matching
    `window.series_hash`) and within the short `window.unmatched_cutoff` backstop, so a newly-added series' files
    are parsed again.
    """

    parse = row.get("parse")
    if not is_json_obj(parse) or not parse.keys() >= _PARSE_FIELDS:
        return False
    stamped = cast("dict[str, Any]", row)
    if "series_fp" not in row:
        return stamp_is_fresh(stamped, window.matched_cutoff)
    return row["series_fp"] == window.series_hash and stamp_is_fresh(stamped, window.unmatched_cutoff)


class ParseCacheRows:
    """The parse cache's rows: fresh-only reads and shape-owning writes over `cache_store`.

    Bound once on `ArrRunDeps` and shared by the planner's sweep and the pending import
    update builder, so the two read one parse per file and a stale row is absent for both.
    """

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def is_fresh(self, filename: str, *, window: ParseFreshness) -> bool:
        """Whether a fresh row exists, by stamp alone (the cache warm-up needs no model)."""

        row = self._store.get_sonarr_parse(filename)
        return row is not None and _parse_is_fresh(row, window=window)

    def read(self, filename: str, *, window: ParseFreshness) -> ParsedFileInfo | None:
        """A fresh, readable row's parse, else None (stale, legacy, malformed, or absent)."""

        row = self._store.get_sonarr_parse(filename)
        if row is None or not _parse_is_fresh(row, window=window):
            return None
        return parse_from_json(row)

    def write(self, filename: str, info: ParsedFileInfo, *, window: ParseFreshness) -> None:
        """Upsert one parse. An UNMATCHED parse (no matched episodes) is pinned to the series hash."""

        row: ParseCacheRow = {"fetched_at": window.now_str, "parse": parse_to_json(info)}
        if not info.matched_episodes:
            row["series_fp"] = window.series_hash
        self._store.put_sonarr_parse(filename, cast("dict[str, Any]", row))
