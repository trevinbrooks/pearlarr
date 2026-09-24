"""The one record timestamp format: what the cache stamps, the pending records age by, and both parse."""

from datetime import datetime, timedelta

UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Timestamp format for cache record fields (`updated_at`, `fetched_at`, `added_at`, `claimed_at`)."""


def stamp_of(moment: datetime) -> str:
    """`moment` in `UPDATED_AT_STR_FORMAT`."""

    return moment.strftime(UPDATED_AT_STR_FORMAT)


def now_stamp() -> str:
    """The current local time in `UPDATED_AT_STR_FORMAT`."""

    return stamp_of(datetime.now())


def parse_stamp(stamp: str) -> datetime:
    """`stamp` parsed back from `UPDATED_AT_STR_FORMAT`, raising like `strptime` on junk."""

    return datetime.strptime(stamp, UPDATED_AT_STR_FORMAT)


def parse_stamp_or_none(stamp: str) -> datetime | None:
    """`stamp` parsed back from `UPDATED_AT_STR_FORMAT`, or None for junk (a raw row's non-string too)."""

    try:
        return parse_stamp(stamp)
    except (TypeError, ValueError):
        return None


def pending_cutoff(max_age_days: int) -> datetime:
    """The oldest clock still inside `imports.pending_max_age_days`: a record's newest claim ages against it."""

    return datetime.now() - timedelta(days=max_age_days)
