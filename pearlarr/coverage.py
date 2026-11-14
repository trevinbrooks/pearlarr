"""Pure episode-coverage formatting helpers."""

from collections.abc import Iterable

from .arr_models import SonarrEpisode
from .releases import EpisodeKey


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


def coverage_of_episodes(episodes: Iterable[SonarrEpisode]) -> str:
    """The `coverage_string` of Sonarr episodes, leaving out any Sonarr sent without its numbers."""

    return coverage_string(
        EpisodeKey(ep.season_number, ep.episode_number)
        for ep in episodes
        if ep.season_number is not None and ep.episode_number is not None
    )
