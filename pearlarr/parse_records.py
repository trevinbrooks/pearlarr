"""The Sonarr parse-cache leaf: one file name's whole `/parse`, its freshness, and the unmatched self-heal.

`ParseRecords` is bound once on `RunDeps`, so the planner's sweep and the seed
builder read one `ParsedFileInfo` per file under one freshness rule.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, NamedTuple, NotRequired, Self, TypedDict, cast

from pydantic import ValidationError

from .cache import AbstractCacheStore, stamp_is_fresh
from .json_narrow import is_json_obj
from .seadex_types import Json, ParsedFileInfo
from .stamps import stamp_of

# How long a persisted Sonarr /parse result stays usable before it's re-queried.
# A filename's season/episode mapping is stable, but Sonarr's /parse depends on
# the current library, so a wrong-but-non-empty match could otherwise be trusted
# forever. Re-validate monthly so such an entry self-heals.
SONARR_PARSE_CACHE_TTL_DAYS = 30

# How long an UNMATCHED parse (Sonarr matched no series) stays usable. Pinned to
# the series-id set (see sonarr_episodes.sonarr_series_fingerprint) so adding the
# series self-heals it. This TTL is only a backstop, so it is short.
SONARR_PARSE_UNMATCHED_TTL_DAYS = 7


class ParseWindow(NamedTuple):
    """The freshness window for one parse pass, computed once per call.

    Bundles the values the parse-cache freshness check and writes thread
    together. One window is opened per pass (the planner's sweep, the seed
    build) and passed down to `ParseRecords`.
    """

    now_str: str
    """Stamps new records."""

    matched_cutoff: datetime
    """Bounds a matched record's TTL."""

    unmatched_cutoff: datetime
    """The short unmatched-record backstop."""

    series_fp: str
    """Pins unmatched records to the current series-id set (so a newly-added series self-heals)."""

    @classmethod
    def open(cls, series_fp: str) -> Self:
        """The window anchored to one instant (never per file)."""

        now = datetime.now()
        return cls(
            now_str=stamp_of(now),
            matched_cutoff=now - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS),
            unmatched_cutoff=now - timedelta(days=SONARR_PARSE_UNMATCHED_TTL_DAYS),
            series_fp=series_fp,
        )


class SonarrParseRecord(TypedDict):
    """One persisted Sonarr `/parse` cache record, keyed by filename."""

    fetched_at: str
    """Stamps the record for TTL eviction."""

    parse: dict[str, Json]
    """The whole `ParsedFileInfo` (see `to_parse_record`): the name's own numbers plus the matched pairs."""

    series_fp: NotRequired[str]
    """`NotRequired` because only an UNMATCHED record carries it (pinning it to the series-id set), and the
    freshness reader dispatches on exactly that presence."""


def to_parse_record(info: ParsedFileInfo) -> dict[str, Json]:
    """The JSON the cache persists for one parse: every field but `offline`.

    The matched pairs' ids ride along so a seed cross-checks them exactly as the
    import poll's live parse does. A series re-add renumbers them, which only
    weakens the seed until the row expires (the poll places from a live parse).
    An offline stand-in is never persisted.
    """

    if info.offline:
        raise ValueError("an offline SxxExx stand-in is never a cache record")
    return info.model_dump(mode="json", exclude={"offline"})


def parsed_info(record: Mapping[str, object]) -> ParsedFileInfo | None:
    """Rebuild a record's parse (the inverse of `to_parse_record`), or None for a legacy or malformed row.

    Validation IS the version gate: a row this build cannot read is a miss and re-fetches.
    """

    raw = record.get("parse")
    if not is_json_obj(raw):
        return None
    try:
        return ParsedFileInfo.model_validate(raw)
    except ValidationError:
        return None


def _parse_is_fresh(record: Mapping[str, object], *, window: ParseWindow) -> bool:
    """True if a persisted parse record is still usable.

    A legacy row (no `parse`) is stale. MATCHED (no `series_fp`): valid for the
    30-day `window.matched_cutoff`. UNMATCHED: valid only while the series-id set
    is unchanged (matching `window.series_fp`) and within the short
    `window.unmatched_cutoff` backstop, so a newly-added series self-heals.
    """

    if "parse" not in record:
        return False
    stamped = cast("dict[str, Any]", record)
    if "series_fp" not in record:
        return stamp_is_fresh(stamped, window.matched_cutoff)
    return record["series_fp"] == window.series_fp and stamp_is_fresh(stamped, window.unmatched_cutoff)


class ParseRecords:
    """The parse-cache leaf: fresh-only reads and shape-owning writes over `cache_store`.

    Bound once on `RunDeps` and shared by the planner's sweep and the seed
    builder, so the two read one parse per file and a stale row is a miss for both.
    """

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def is_fresh(self, filename: str, *, window: ParseWindow) -> bool:
        """Whether a fresh record exists, by stamp alone (the warm pass needs no model)."""

        record = self._store.get_sonarr_parse(filename)
        return record is not None and _parse_is_fresh(record, window=window)

    def read(self, filename: str, *, window: ParseWindow) -> ParsedFileInfo | None:
        """A fresh, readable record's parse, else None (stale, legacy, malformed, or absent)."""

        record = self._store.get_sonarr_parse(filename)
        if record is None or not _parse_is_fresh(record, window=window):
            return None
        return parsed_info(record)

    def write(self, filename: str, info: ParsedFileInfo, *, window: ParseWindow) -> None:
        """Upsert one parse. An UNMATCHED parse (no matched pairs) is pinned to the series fingerprint."""

        record: SonarrParseRecord = {"fetched_at": window.now_str, "parse": to_parse_record(info)}
        if not info.matched_episodes:
            record["series_fp"] = window.series_fp
        self._store.put_sonarr_parse(filename, cast("dict[str, Any]", record))
