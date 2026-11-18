"""Pure episode-coverage formatting helpers."""

from collections.abc import Iterable

from .aired_view import printed_key
from .arr_models import SonarrEpisode
from .releases import EpisodeKey, PrintedKey, aired_label


def format_episode_ranges(episode_numbers: Iterable[int]) -> str:
    """Condense one season's episode numbers into a readable range string.

    Contiguous ranges are collapsed (e.g. [1, 2, 3] -> "E01-E03"), lone episodes
    are kept as-is (e.g. [5] -> "E05"), and gaps split into multiple comma-separated ranges (e.g. [1, 2, 3, 7, 8] ->
    "E01-E03, E07-E08").
    """

    episodes = sorted(set(episode_numbers))
    if not episodes:
        return ""

    # Walk the sorted episodes, breaking into ranges wherever they aren't
    # consecutive
    runs: list[tuple[int, int]] = []
    run_start = run_end = episodes[0]
    for episode in episodes[1:]:
        if episode == run_end + 1:
            run_end = episode
        else:
            runs.append((run_start, run_end))
            run_start = run_end = episode
    runs.append((run_start, run_end))

    return ", ".join(f"E{start:02d}" if start == end else f"E{start:02d}-E{end:02d}" for start, end in runs)


def coverage_string(keys: Iterable[EpisodeKey]) -> str:
    """One-line season and episode coverage, as "S04 E01-E12" or "S00 E10, S02 E01-E12".

    The seasons run in order, each with its episodes condensed by `format_episode_ranges`. "" when there are no
    keys (a Radarr movie, or a release none of whose files placed), so callers can treat it as "URL only".
    """

    episodes_by_season: dict[int, set[int]] = {}
    for key in keys:
        episodes_by_season.setdefault(key.season, set()).add(key.episode)
    return ", ".join(
        f"S{season:02d} {format_episode_ranges(episodes_by_season[season])}" for season in sorted(episodes_by_season)
    )


def printed_coverage(keys: Iterable[PrintedKey]) -> str:
    """One-line coverage of episodes as lines print them, as "S01 E01-E12, aired S02 E13".

    The numbers Sonarr lists come first, as `coverage_string` gives them. The aired numbers of episodes Sonarr has
    no row for follow, as their own `coverage_string` behind the one word "aired". "" when there are no keys.
    """

    listed: list[EpisodeKey] = []
    aired: list[EpisodeKey] = []
    for printed in keys:
        (aired if printed.aired else listed).append(printed.key)
    parts = [coverage_string(listed)]
    if aired:
        parts.append(aired_label(coverage_string(aired)))
    return ", ".join(part for part in parts if part)


def coverage_of_episodes(episodes: Iterable[SonarrEpisode]) -> str:
    """The `printed_coverage` of Sonarr episodes (`printed_key`), leaving out any with no season or episode number."""

    return printed_coverage(
        printed_key(ep) for ep in episodes if ep.season_number is not None and ep.episode_number is not None
    )
