"""The download-decision engine: which SeaDex releases to grab.

The pure set-logic helpers (``normalize_rg`` .. ``get_all_seadex_rgs_per_episode``)
were relocated here verbatim in Phase 1; the ``DownloadPlanner`` class that
consumes them — the heart of the app — landed in Phase 4 (see ``REFACTOR_PLAN.md``).

``DownloadPlanner`` is near-pure: it consumes the shaped ``seadex_dict``, the
Arr's current release info, an optional episode list, and the cached torrent
hashes, and returns a :class:`PlanResult`. It flips the per-url ``download``
flags in place and reports *what to log* (``skip_notices``) and *what was skipped
for being private-only* (``public_only_*``) as data, rather than reaching into
the orchestrator's run state or its log formatter.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import compress

from .log import indent_string


@dataclass
class SkipNotice:
    """A release skipped purely for being private, for the caller to log.

    Rendered by the orchestrator as ``"<groups> <reason>"`` on a ``skipped``
    detail line, replacing the inline ``log_fmt`` call this used to make from
    deep inside the decision engine.
    """

    groups: list[str]
    reason: str
    level: int = logging.WARNING


@dataclass
class PublicOnlySkips:
    """The private-only skip outcome of ``reduce_overlapping_downloads``.

    ``skipped`` is True when at least one set of same-files release groups was
    dropped because, with ``public_only`` on, none were available publicly;
    ``groups`` names them (for the run summary) and ``notices`` is what to log.
    """

    skipped: bool = False
    groups: list[str] = field(default_factory=list)
    notices: list[SkipNotice] = field(default_factory=list)


@dataclass
class PlanResult:
    """The download-decision engine's output.

    ``seadex_dict`` is the same dict passed in, annotated in place with per-url
    ``download`` flags. ``torrent_hashes`` is the unique set to remember in the
    cache record. The remaining fields surface the private-only skip outcome so
    the orchestrator can log it, name it in the summary, and decide whether to
    cache the title as done.
    """

    seadex_dict: dict
    torrent_hashes: list[str]
    public_only_skipped: bool = False
    public_only_groups: list[str] = field(default_factory=list)
    skip_notices: list[SkipNotice] = field(default_factory=list)


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


class DownloadPlanner:
    """Decides which SeaDex releases to grab for one AniList entry.

    Constructed once per run with the three config flags it consults; every
    decision method takes the already-shaped ``seadex_dict`` plus the Arr's
    release info as arguments and returns a :class:`PlanResult`. The planner
    keeps a logger only for the per-release debug breadcrumbs; the user-facing
    private-only skip is returned as a :class:`SkipNotice`, never logged here.
    """

    def __init__(
        self,
        *,
        public_only: bool,
        interactive: bool,
        use_torrent_hash_to_filter: bool,
        logger: logging.Logger,
    ) -> None:
        self.public_only = public_only
        self.interactive = interactive
        self.use_torrent_hash_to_filter = use_torrent_hash_to_filter
        self.logger = logger

    def plan(
        self,
        *,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        cached_hashes: list,
        ep_list: list | None = None,
    ) -> PlanResult:
        """Flip the download flags and return the full plan for an entry.

        Selects the hash-based or release-group-based strategy from the config
        flag, then unions in the cached hashes (release-group path only — the
        hash path already lists every url's hash) and de-duplicates.

        Args:
            seadex_dict: Dictionary of SeaDex releases (annotated in place)
            arr: Type of arr instance
            arr_release_dict: Dictionary of arr release properties
            cached_hashes: Torrent hashes already remembered for this entry
            ep_list: List of episodes. Defaults to None
        """

        if self.use_torrent_hash_to_filter:
            result = self.filter_by_torrent_hash(
                seadex_dict=seadex_dict,
                cached_hashes=cached_hashes,
            )
        else:
            result = self.filter_by_release_group(
                seadex_dict=seadex_dict,
                arr=arr,
                arr_release_dict=arr_release_dict,
                ep_list=ep_list,
            )

            # Also include any cached hashes
            result.torrent_hashes.extend(cached_hashes)

        # Make sure the hashes are unique
        result.torrent_hashes = list(set(result.torrent_hashes))

        return result

    def filter_by_torrent_hash(
        self,
        seadex_dict: dict,
        cached_hashes: list,
    ) -> PlanResult:
        """Select downloads if the torrent hash is not already in the cache

        Multiple "best" releases are all grabbed, except where several cover
        the same files (see reduce_overlapping_downloads), in which case only
        one is kept

        Args:
            seadex_dict: Dictionary of SeaDex releases
            cached_hashes: Torrent hashes already remembered for this entry
        """

        torrent_hashes = []

        for seadex_rg, seadex_rg_item in seadex_dict.items():

            self.logger.debug(
                indent_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.get("urls", {})
            for url_item in seadex_urls.values():

                url_hash = url_item.get("hash", None)

                # If the URL is already in the hash cache, then append but don't set to download
                torrent_hashes.append(url_hash)
                if url_hash not in cached_hashes:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {url_hash} not found in cache. "
                            f"Will add to downloads",
                        ),
                    )

                    url_item.update({"download": True})

                else:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {url_hash} in cache. Will skip download",
                        ),
                    )

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        skips = self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            public_only_skipped=skips.skipped,
            public_only_groups=skips.groups,
            skip_notices=skips.notices,
        )

    def filter_by_release_group(
        self,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        ep_list: list | None = None,
    ) -> PlanResult:
        """Filter torrents by release group

        This is either an episode-by-episode for the Sonarr
        case where we can parse episodes, or a more blunt
        hammer just checking against anything for Radarr
        and weirdly named TV

        Args:
            seadex_dict: Dictionary of SeaDex releases
            arr: Type of arr instance
            arr_release_dict: Dictionary of arr release properties
            ep_list: List of episodes. Defaults to None
        """

        # The release-group names, used both for display (insertion order
        # preserved) and for membership tests below. A dict keys view already
        # supports `in` in O(1), so there's no need to materialize a list.
        arr_release_groups = arr_release_dict.keys()

        # And also just check if any release group matches
        # any Arr release tag
        seadex_keys = set(seadex_dict.keys())
        overlapping_results = any(rg in seadex_keys for rg in arr_release_groups)

        # Index the Sonarr episodes by (season, episode) once, shared by both
        # the overlap map below and the per-episode match loop: looking up a
        # parsed SeaDex (season, episode) is then an O(1) dict op rather than a
        # fresh scan of the whole list. First entry wins on a duplicate key
        # (Sonarr episodes are unique by season+episode).
        sonarr_by_key: dict = {}
        for sonarr_ep in ep_list or []:
            sonarr_by_key.setdefault(
                (
                    sonarr_ep.get("seasonNumber", 999),
                    sonarr_ep.get("episodeNumber", 999),
                ),
                sonarr_ep,
            )

        # If we have overlaps, get a note of them here, reusing the index above
        all_seadex_rgs_per_episode = get_all_seadex_rgs_per_episode(
            seadex_dict=seadex_dict,
            sonarr_by_key=sonarr_by_key,
        )

        # Resolve once: the per-episode debug lines below sit in the hot
        # matching loop, so this lets us skip building their f-strings on a
        # normal INFO run instead of formatting them only to discard them.
        debug_on = self.logger.isEnabledFor(logging.DEBUG)

        for seadex_rg, seadex_rg_item in seadex_dict.items():

            self.logger.debug(
                indent_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.get("urls", {})
            for url, url_item in seadex_urls.items():

                seadex_episodes = url_item.get("episodes", [])

                # Simple case, we have no episode mappings, so
                # just fall back to checking against release group
                if not seadex_episodes:
                    self._match_url_no_episodes(
                        seadex_rg=seadex_rg,
                        url=url,
                        url_item=url_item,
                        arr=arr,
                        arr_release_dict=arr_release_dict,
                        arr_release_groups=arr_release_groups,
                        overlapping_results=overlapping_results,
                    )
                    continue

                self._match_url_episodes(
                    seadex_rg=seadex_rg,
                    url=url,
                    url_item=url_item,
                    arr=arr,
                    seadex_episodes=seadex_episodes,
                    sonarr_by_key=sonarr_by_key,
                    all_seadex_rgs_per_episode=all_seadex_rgs_per_episode,
                    has_ep_list=ep_list is not None,
                    debug_on=debug_on,
                )

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        skips = self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        # Build the hash list from whatever is still flagged for download, so it
        # always matches the exact set of torrents we'll add. Private torrents
        # have no infohash, so skip those
        torrent_hashes = [
            url_item["hash"]
            for rg_item in seadex_dict.values()
            for url_item in rg_item.get("urls", {}).values()
            if url_item.get("download", False) and url_item.get("hash") is not None
        ]

        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            public_only_skipped=skips.skipped,
            public_only_groups=skips.groups,
            skip_notices=skips.notices,
        )

    def _match_url_no_episodes(
        self,
        *,
        seadex_rg: str,
        url: str,
        url_item: dict,
        arr: str,
        arr_release_dict: dict,
        arr_release_groups: Iterable[str],
        overlapping_results: bool,
    ) -> None:
        """Decide a single url with no parsed episodes, by release group + size.

        Flips ``url_item["download"]`` in place. The blunt fallback used for
        Radarr and weirdly named TV: if the group isn't in the Arr's releases
        (and nothing overlaps) grab it; if it is, grab it only when the file
        sizes are disjoint.
        """

        if seadex_rg not in arr_release_groups and not overlapping_results:
            self.logger.debug(
                indent_string(
                    f"SeaDex release group {seadex_rg} not in {arr.capitalize()} releases: "
                    f"{', '.join([str(x) for x in arr_release_groups])} - will download {url}",
                ),
            )

            url_item.update({"download": True})

        # If the group matches, fall through to a size comparison
        if seadex_rg in arr_release_groups:

            seadex_file_sizes = url_item.get("size", [])
            arr_file_sizes = arr_release_dict[seadex_rg].get("size", [])

            if not isinstance(arr_file_sizes, list):
                arr_file_sizes = [arr_file_sizes]

            # If we have no overlaps at all, then add
            if set(seadex_file_sizes).isdisjoint(arr_file_sizes):
                self.logger.debug(
                    indent_string(
                        f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                        f"{', '.join([str(x) for x in arr_release_groups])}, but file sizes do not match - will download {url}",
                    ),
                )

                url_item.update({"download": True})

            else:
                self.logger.debug(
                    indent_string(
                        f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                        f"{', '.join([str(x) for x in arr_release_groups])}, and file sizes match",
                    ),
                )

    def _match_url_episodes(
        self,
        *,
        seadex_rg: str,
        url: str,
        url_item: dict,
        arr: str,
        seadex_episodes: list,
        sonarr_by_key: dict,
        all_seadex_rgs_per_episode: dict,
        has_ep_list: bool,
        debug_on: bool,
    ) -> None:
        """Decide a single url against its parsed episodes, per episode.

        Flips ``url_item["download"]`` in place. For each parsed SeaDex episode
        we check whether it exists in the Sonarr index, whether the release
        group matches, and whether the file sizes match; a release-group
        mismatch with no covering alternative, or an all-sizes mismatch among
        the rg-matched episodes, flips download on.
        """

        # At this point, we need an episode list from Sonarr. A non-None but
        # empty list still runs the (no-op) loop below; only an absent list skips.
        if not has_ep_list:
            self.logger.debug(
                "Skipping per-episode check: no Sonarr episode list available",
            )
            return

        # For each episode we've parsed from the torrent, check if a) it exists in the Sonarr list, b) if
        # the release group matches, and c) if the file sizes match. If there's any mismatch between release
        # groups (and there are no alternatives), then flip download to True. If all the sizes mismatch,
        # flip download to true

        rg_matches = [False] * len(seadex_episodes)
        size_matches = [False] * len(seadex_episodes)

        for seadex_idx, seadex_ep in enumerate(seadex_episodes):

            seadex_ep_season = seadex_ep.get("season", 888)
            seadex_ep_episode = seadex_ep.get("episode", 888)
            seadex_ep_size = seadex_ep.get("size", None)

            # O(1) lookup into the indexed Sonarr episodes instead of
            # re-scanning the whole list for every parsed episode
            sonarr_ep = sonarr_by_key.get(
                (seadex_ep_season, seadex_ep_episode),
            )
            if sonarr_ep is None:
                continue

            # Get the matched Sonarr episode's file size
            sonarr_ep_size = sonarr_ep.get("episodeFile", {}).get(
                "size", None,
            )

            # Do the sizes match? A missing Sonarr file reports no
            # size, so guard against None == None reading as a match
            # when neither side actually has a size.
            size_match = (
                sonarr_ep_size is not None
                and sonarr_ep_size == seadex_ep_size
            )

            season_ep_str = (
                f"S{seadex_ep_season:02d}E{seadex_ep_episode:02d}"
            )

            # Check SeaDex release group matches the episode release group in Sonarr
            sonarr_rg = sonarr_ep.get("episodeFile", {}).get(
                "releaseGroup", None,
            )
            sonarr_rg_normalized = normalize_rg(sonarr_rg)
            seadex_rg_normalized = normalize_rg(seadex_rg)
            # If not, flag as should be downloaded if it's not
            # already in some overlapping release.
            # normalized name indexes all_seadex_rgs_per_episode, so compare the normalized name
            if (
                sonarr_rg_normalized != seadex_rg_normalized
                and sonarr_rg_normalized
                not in all_seadex_rgs_per_episode["all"]
            ):

                # Avoid duplicating when another release already covers it
                all_seadex_rg = all_seadex_rgs_per_episode.get(
                    season_ep_str, (),
                )

                if sonarr_rg_normalized not in all_seadex_rg:
                    if debug_on:
                        self.logger.debug(
                            indent_string(
                                f"SeaDex release group {seadex_rg} differs from "
                                f"{arr.capitalize()} release for "
                                f"{season_ep_str} ({sonarr_rg}) and no other "
                                f"recommended release covers it - will download {url}",
                            ),
                        )

                    url_item.update({"download": True})

            else:

                if debug_on:
                    self.logger.debug(
                        indent_string(
                            f"Found SeaDex match to {arr.capitalize()} "
                            f"for {season_ep_str}.",
                        ),
                    )
                    if not size_match:
                        self.logger.debug(
                            indent_string(
                                f"-> Sizes are different: "
                                f"{sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
                            ),
                        )
                    else:
                        self.logger.debug(
                            indent_string(
                                f"-> Sizes match: {sonarr_ep_size}",
                            ),
                        )

                rg_matches[seadex_idx] = True

            # Now check against file size
            if size_match:
                size_matches[seadex_idx] = True

        # If we have matched the release groups but not the file sizes, then flag that
        # here and mark for download
        size_matches = list(compress(size_matches, rg_matches))
        if size_matches and not any(size_matches):
            self.logger.debug(
                indent_string(
                    f"File sizes all differ for release group {seadex_rg} - will download {url}",
                ),
            )
            url_item.update({"download": True})

    def reduce_overlapping_downloads(
        self,
        seadex_dict: dict,
    ) -> PublicOnlySkips:
        """Reduce overlapping flagged downloads down to a single release group

        Where multiple preferred release groups cover the same files and the
        Arr doesn't already have any of them, we only want to grab one. If
        public_only is set, we prefer a public release group and drop the
        private ones. If the only options are private, we record a SkipNotice
        and skip the title (without caching it as done) rather than grabbing a
        private release.

        Mutates the download flags on seadex_dict in place and returns the
        private-only skip outcome (skipped flag, group names, notices to log).
        Skipped entirely in interactive mode, where the user has already
        hand-picked what to grab.

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        skips = PublicOnlySkips()

        # In interactive mode the user has explicitly chosen which releases to
        # grab, so don't second-guess them by dropping any
        if self.interactive:
            return skips

        def is_flagged(rg_item: dict) -> bool:
            return any(
                u.get("download", False) for u in rg_item.get("urls", {}).values()
            )

        def is_public_group(rg_item: dict) -> bool:
            return any(
                u.get("is_public", False) for u in rg_item.get("urls", {}).values()
            )

        def unflag(rg_item: dict) -> None:
            for u in rg_item.get("urls", {}).values():
                u["download"] = False

        same_files_groups = get_same_files_groups(seadex_dict)

        for same_files in same_files_groups:

            # Only the release groups the Arr doesn't already have are flagged
            flagged = [rg for rg in same_files if is_flagged(seadex_dict[rg])]
            if len(flagged) == 0:
                continue

            if self.public_only:
                public_flagged = [
                    rg for rg in flagged if is_public_group(seadex_dict[rg])
                ]

                if len(public_flagged) == 0:
                    # The Arr has none of these release groups, public_only is
                    # set, but none are available on a public tracker. Don't
                    # grab a private release, just record a skip notice and skip.
                    # Flag the skip so the caller doesn't cache the title as done
                    skips.notices.append(
                        SkipNotice(
                            groups=list(flagged),
                            reason="private-only (public_only on)",
                            level=logging.WARNING,
                        ),
                    )
                    skips.skipped = True
                    skips.groups.extend(flagged)
                    for rg in flagged:
                        unflag(seadex_dict[rg])
                    continue

                # Keep the first public release group, drop everything else
                keeper = public_flagged[0]
            else:
                # We don't care about public/private, just keep the first one
                keeper = flagged[0]

            for rg in flagged:
                if rg == keeper:
                    continue

                self.logger.debug(
                    indent_string(
                        f"Not downloading release group {rg}: release group "
                        f"{keeper} already covers the same files",
                    ),
                )
                unflag(seadex_dict[rg])

        return skips

    @staticmethod
    def get_any_to_download(seadex_dict: dict) -> bool:
        """Check if any torrents are marked as to download

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        return any(
            url_item.get("download", False)
            for rg_item in seadex_dict.values()
            for url_item in rg_item.get("urls", {}).values()
        )
