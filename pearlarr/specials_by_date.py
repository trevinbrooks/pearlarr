"""Picks a title's specials by AniList's air dates and episode length when the mapped ones aired on other days."""

from collections.abc import Iterable, Sequence
from datetime import date, timedelta

from .seadex_types import AniListAiring, SonarrEpisode

# How far the mapped specials may air from AniList's dates and still agree with them. The two sources often date one
# special days apart, and a weekly run one airing apart.
_AGREES_WITHIN = timedelta(days=14)

# How far outside AniList's dates a special may air and still be picked. The sources date one airing a day apart
# when it ran late at night.
_PICKED_WITHIN = timedelta(days=1)

# A runtime is close to AniList's episode length within this many minutes, or this share of the length if that's more.
# Both sources round to the minute, and a film's runtime differs by cut.
_CLOSE_MINUTES = 3
_CLOSE_SHARE = 0.2


def specials_by_date(
    mapped: Sequence[SonarrEpisode], series_episodes: Iterable[SonarrEpisode], airing: AniListAiring
) -> list[SonarrEpisode] | None:
    """The specials that replace `mapped`, a title's mapped episodes, or None when the mapping stands.

    It stands unless every mapped episode is a special with an air date, `airing` has a start date, an episode
    count and an episode length, and the mapped specials miss its dates: the earliest aired more than two weeks
    from the start date, or the latest more than two weeks from the end date (the start date when there is none).

    The replacements come from the specials in `series_episodes`, the title's whole series, whose runtime is close
    to the episode length (`_runtime_close`). Those aired between the two dates, give or take a day, replace the
    mapping when there are exactly as many as the episode count. Failing that, for two episodes with the end date
    after the start date, the one special aired within a day of the start date and the one within a day of the end
    date do. Specials that are already the mapped ones replace nothing. The result is in `series_episodes` order.
    """

    start, count, duration = airing.start, airing.episodes, airing.duration
    if start is None or not count or not duration:
        return None
    if not mapped or any(ep.season_number != 0 for ep in mapped):
        return None
    aired = sorted(ep.air_date for ep in mapped if ep.air_date is not None)
    if len(aired) != len(mapped):
        return None
    end = airing.end or start
    if abs(aired[0] - start) <= _AGREES_WITHIN and abs(aired[-1] - end) <= _AGREES_WITHIN:
        return None

    specials = [ep for ep in series_episodes if ep.season_number == 0 and _runtime_close(ep.runtime, duration)]
    between = [ep for ep in specials if _aired_within(ep, start - _PICKED_WITHIN, end + _PICKED_WITHIN)]
    if len(between) == count and {ep.id for ep in between} != {ep.id for ep in mapped}:
        return between
    if count == 2 and start < end:
        firsts = [ep for ep in specials if _aired_within(ep, start - _PICKED_WITHIN, start + _PICKED_WITHIN)]
        lasts = [ep for ep in specials if _aired_within(ep, end - _PICKED_WITHIN, end + _PICKED_WITHIN)]
        # The mapped pair is never this one: a pair on the two dates agrees with them, and stood above.
        if len(firsts) == 1 and len(lasts) == 1 and firsts[0].id != lasts[0].id:
            pair = {firsts[0].id, lasts[0].id}
            return [ep for ep in specials if ep.id in pair]
    return None


def _runtime_close(runtime: int | None, duration: int) -> bool:
    """Whether a special's runtime is close to AniList's episode length. A missing or zero runtime never is."""

    return (
        runtime is not None and runtime > 0 and abs(runtime - duration) <= max(_CLOSE_MINUTES, _CLOSE_SHARE * duration)
    )


def _aired_within(episode: SonarrEpisode, first: date, last: date) -> bool:
    """Whether the episode aired from `first` to `last`, both days included. One with no air date never did."""

    return episode.air_date is not None and first <= episode.air_date <= last
