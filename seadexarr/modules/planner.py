"""Release-matching set logic for the download-decision engine.

Extracted verbatim from ``seadex_arr.py`` in the Phase 1 decomposition (see
``REFACTOR_PLAN.md``). These pure helpers compare what the Arr already has
against what SeaDex recommends. The ``DownloadPlanner`` class that will consume
them (and the rest of the decision engine) lands here in Phase 4.
"""

from collections.abc import Iterable


def normalize_rg(name: str | None) -> str | None:
    """Normalize a release group name for comparison

    Lower-cases and strips surrounding whitespace and dashes so that the same
    group named slightly differently by Sonarr and SeaDex (e.g. "Era-Raws" vs.
    "era-raws ") compare equal. Returns None for a missing/blank name.

    Args:
        name (str | None): Release group name
    """

    if not name:
        return None
    return name.strip().strip("-").casefold()


def get_episode_keys(all_episodes: Iterable[dict]) -> set:
    """Build the set of (season, episode) keys an episode list covers

    Reduces a release's parsed episode list to the set of (season, episode)
    pairs it contains, so different SeaDex release groups can be compared by
    what files they cover.

    Args:
        all_episodes (iterable): Parsed episode dicts with "season"/"episode"
    """

    return {(ep.get("season"), ep.get("episode")) for ep in all_episodes}


def get_same_files_groups(seadex_dict: dict) -> list:
    """Group SeaDex release groups that cover exactly the same files

    Release groups are grouped by their parsed episode coverage: two groups are
    only treated as covering the same files when their parsed episode lists are
    identical. This is deliberately stricter than "episodes overlap" -- groups
    that overlap without being equal (e.g., a full-season batch and a single
    cour) cover *different* files and must not be collapsed, or we'd silently
    drop episodes when keeping only one of them.

    Release groups with no episode parsing at all (e.g., Radarr movies) are
    treated as covering the same files. Release groups whose files couldn't be
    parsed (Sonarr parse failure, empty episode list) are each kept on their
    own: we can't prove what they cover, so we'd rather grab a duplicate than
    silently drop content. Returns a list of lists of release group names.

    Args:
        seadex_dict (dict): Dictionary of SeaDex releases
    """

    grouped = {}
    for rg, rg_item in seadex_dict.items():
        all_episodes = rg_item.get("all_episodes", None)

        if all_episodes is None:
            # No episode parsing for this Arr (e.g., Radarr): treat as one movie
            key = "__no_episode_parsing__"
        elif len(all_episodes) == 0:
            # Parsing ran but found nothing: keep this group on its own so we
            # never drop content we couldn't verify
            key = ("__unparsed__", rg)
        else:
            key = frozenset(get_episode_keys(all_episodes))

        # Insertion-ordered dict preserves first-seen group order for us
        grouped.setdefault(key, []).append(rg)

    return list(grouped.values())


def get_all_seadex_rgs_per_episode(
    seadex_dict: dict,
    sonarr_by_key: dict,
) -> dict:
    """Get a list of all SeaDex releases per-episode

    Args:
        seadex_dict: Dictionary of SeaDex releases
        sonarr_by_key: Sonarr episodes indexed by (season, episode). A parsed
            SeaDex (season, episode) is recorded only when Sonarr has it, which
            this makes an O(1) key lookup. Built once by the caller and shared
            with the per-episode match loop in filter_by_release_group.
    """

    all_seadex_rgs_per_episode: dict[str, set] = {"all": set()}

    if len(seadex_dict) > 1:
        for seadex_rg, seadex_rg_item in seadex_dict.items():

            # Index by the normalized name so the membership checks in
            # filter_by_release_group are case- and dash-insensitive
            seadex_rg_normalized = normalize_rg(seadex_rg)

            seadex_urls = seadex_rg_item.get("urls", {})
            for url_item in seadex_urls.values():

                seadex_episodes = url_item.get("episodes", [])

                # If we haven't managed to parse, then set this up as an
                # "all" episode fallback
                if len(seadex_episodes) == 0:
                    all_seadex_rgs_per_episode["all"].add(seadex_rg_normalized)

                for seadex_ep in seadex_episodes:
                    season = seadex_ep.get("season", 888)
                    episode = seadex_ep.get("episode", 888)

                    # Only record episodes Sonarr actually has, matching the
                    # original per-episode gate against the episode list
                    if (season, episode) in sonarr_by_key:
                        season_key = f"S{season:02d}E{episode:02d}"
                        all_seadex_rgs_per_episode.setdefault(
                            season_key, set(),
                        ).add(seadex_rg_normalized)

    return all_seadex_rgs_per_episode
