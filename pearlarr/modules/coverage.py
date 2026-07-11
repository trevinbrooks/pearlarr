"""Pure episode-coverage formatting helpers."""

from collections.abc import Iterable

from .seadex_types import EpisodeRecord, SonarrEpisode


def format_episode_ranges(episode_numbers: Iterable[int]) -> str:
    """Condense one season's episode numbers into a readable range string.

    Contiguous runs are collapsed (e.g. [1, 2, 3] -> "E01-E03"), lone episodes
    are kept as-is (e.g. [5] -> "E05"), and gaps split into multiple comma-separated ranges (e.g. [1, 2, 3, 7, 8] -> "E01-E03, E07-E08").
    """

    episodes = sorted(set(episode_numbers))
    if not episodes:
        return ""

    # Walk the sorted episodes, breaking into runs wherever they aren't
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


def format_episode_coverage(episodes: list[EpisodeRecord]) -> list[tuple[str, str]] | None:
    """Summarize the Sonarr season/episode coverage of a torrent, per season

    Returns a list of (season_label, episode_ranges) tuples, one per season
    the torrent covers, ordered by season. The season label is e.g. "S01"
    and the episode ranges condense contiguous runs, e.g. "E01-E12" or
    "E01-E03, E07-E12" for a season with a gap, or "E05" for a lone episode.

    Returns None when there is no parsed episode info (e.g., Radarr movies,
    or a Sonarr parse failure).

    Args:
        episodes: The coverage records parsed onto each torrent's url_item
    """

    if not episodes:
        return None

    # Collect the episode numbers seen for each season
    episodes_by_season: dict[int, set[int]] = {}
    for ep in episodes:
        season = ep.season
        episode = ep.episode
        if season is None or episode is None:
            continue
        episodes_by_season.setdefault(season, set()).add(episode)

    if not episodes_by_season:
        return None

    return [
        (f"S{season:02d}", format_episode_ranges(episodes_by_season[season])) for season in sorted(episodes_by_season)
    ]


def coverage_string(episodes: list[EpisodeRecord]) -> str:
    """One-line season/episode coverage, e.g. "S04 E01-E12" or "S00 E10, S02 E01-E12".

    Returns "" when there's no parsed episode info (e.g., a Radarr movie), so
    callers can treat it as "URL only".
    """

    coverage = format_episode_coverage(episodes)
    if not coverage:
        return ""
    return ", ".join(f"{label} {ranges}" for label, ranges in coverage)


def episodes_from_ep_list(
    ep_list: list[SonarrEpisode] | None,
    missing_only: bool = False,
) -> list[EpisodeRecord]:
    """Convert a Sonarr ep_list into `EpisodeRecord` coverage records.

    Sonarr episodes carry "seasonNumber"/"episodeNumber"; the coverage helpers
    read `EpisodeRecord.season`/`.episode`. `missing_only` keeps only missing
    episodes (no file on disk), to summarize what is still needed.
    """

    episodes: list[EpisodeRecord] = []
    for ep in ep_list or []:
        if missing_only and ep.episode_file_id != 0:
            continue
        episodes.append(
            EpisodeRecord(
                season=ep.season_number,
                episode=ep.episode_number,
            ),
        )
    return episodes
