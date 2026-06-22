"""Pure episode-coverage formatting helpers.

Extracted verbatim from ``seadex_arr.py`` in the Phase 1 decomposition (see
``REFACTOR_PLAN.md``). These functions are pure -- they format Sonarr / SeaDex
episode data for display and carry no I/O or shared state. ``SeaDexArr`` keeps
thin delegating methods so subclasses can still call them via ``self``.
"""

from collections.abc import Iterable


def format_episode_ranges(episode_numbers: Iterable[int]) -> str:
    """Condense a set of episode numbers into a readable range string

    Contiguous runs are collapsed (e.g. [1, 2, 3] -> "E01-E03"), lone episodes
    are kept as-is (e.g. [5] -> "E05"), and gaps split into multiple comma-separated ranges (e.g. [1, 2, 3, 7, 8] -> "E01-E03, E07-E08").

    Args:
        episode_numbers (iterable): Episode numbers within a single season
    """

    episodes = sorted(set(episode_numbers))
    if not episodes:
        return ""

    # Walk the sorted episodes, breaking into runs wherever they aren't
    # consecutive
    runs = []
    run_start = run_end = episodes[0]
    for episode in episodes[1:]:
        if episode == run_end + 1:
            run_end = episode
        else:
            runs.append((run_start, run_end))
            run_start = run_end = episode
    runs.append((run_start, run_end))

    return ", ".join(
        f"E{start:02d}" if start == end else f"E{start:02d}-E{end:02d}"
        for start, end in runs
    )


def format_episode_coverage(episodes: list) -> list | None:
    """Summarize the Sonarr season/episode coverage of a torrent, per season

    Returns a list of (season_label, episode_ranges) tuples, one per season
    the torrent covers, ordered by season. The season label is e.g. "S01"
    and the episode ranges condense contiguous runs, e.g. "E01-E12" or
    "E01-E03, E07-E12" for a season with a gap, or "E05" for a lone episode.

    Returns None when there is no parsed episode info (e.g., Radarr movies,
    or a Sonarr parse failure).

    Args:
        episodes (list): List of {"season", "episode", ...} dicts,
            as parsed onto each torrent's url_item
    """

    if not episodes:
        return None

    # Collect the episode numbers seen for each season
    episodes_by_season = {}
    for ep in episodes:
        season = ep.get("season")
        episode = ep.get("episode")
        if season is None or episode is None:
            continue
        episodes_by_season.setdefault(season, set()).add(episode)

    if not episodes_by_season:
        return None

    return [
        (f"S{season:02d}", format_episode_ranges(episodes_by_season[season]))
        for season in sorted(episodes_by_season)
    ]


def coverage_string(episodes: list) -> str:
    """One-line season/episode coverage, e.g. "S04 E01-E12" or
    "S00 E10, S02 E01-E12". Returns "" when there's no parsed episode info
    (e.g., a Radarr movie), so callers can treat it as "URL only".

    Args:
        episodes (list): {"season", "episode"} dicts
    """

    coverage = format_episode_coverage(episodes)
    if not coverage:
        return ""
    return ", ".join(f"{label} {ranges}" for label, ranges in coverage)


def episodes_from_ep_list(ep_list: list | None, missing_only: bool = False) -> list:
    """Convert a Sonarr ep_list into {"season","episode"} coverage dicts

    Sonarr episodes carry "seasonNumber"/"episodeNumber"; the coverage
    helpers expect "season"/"episode". Optionally, keep only missing episodes
    (no file on disk) to summarize what is still needed.

    Args:
        ep_list (list): Sonarr episode dicts
        missing_only (bool): Keep only episodes with no file. Defaults to False
    """

    episodes = []
    for ep in ep_list or []:
        if missing_only and ep.get("episodeFileId", 0) != 0:
            continue
        episodes.append(
            {
                "season": ep.get("seasonNumber"),
                "episode": ep.get("episodeNumber"),
            },
        )
    return episodes
