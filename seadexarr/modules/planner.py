"""The download-decision engine: which SeaDex releases to grab.

``DownloadPlanner`` is near-pure: it consumes the shaped ``seadex_dict``, the
Arr's current release info, an optional episode list, and the cached torrent
hashes, and returns a :class:`PlanResult`. It flips the per-url ``download``
flags in place and reports *what to log* (``skip_notices``) and *what was skipped
for being private-only* (``private_only_*``) as data, rather than reaching into
the orchestrator's run state or its log formatter.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import compress

from .config import Arr
from .log import indent_string
from .manual_import import normalize_group
from .seadex_types import (
    ArrReleaseDict,
    EpisodeRecord,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
    as_size_list,
    season_episode_key,
)


@dataclass
class SkipNotice:
    """A release dropped for being private, for the caller to log.

    Rendered by the orchestrator as ``"<groups> <reason>"`` on a ``skipped``
    detail line, replacing the inline ``log_fmt`` call this used to make from
    deep inside the decision engine. A warn-and-hold skip logs at WARNING; a
    drop covered by a public fallback logs at INFO.
    """

    groups: list[str]
    reason: str
    level: int = logging.WARNING


@dataclass
class PrivateOnlySkips:
    """The private-only skip outcome of ``reduce_overlapping_downloads``.

    ``skipped`` is True when at least one set of same-files release groups was
    dropped because none were available publicly and no public fallback
    covered the same files; ``groups`` names them (for the run summary) and
    ``notices`` is what to log.
    """

    skipped: bool = False
    groups: list[str] = field(default_factory=list[str])
    notices: list[SkipNotice] = field(default_factory=list[SkipNotice])


@dataclass
class PlanResult:
    """The download-decision engine's output.

    ``seadex_dict`` is the same dict passed in, annotated in place with per-url
    ``download`` flags. ``torrent_hashes`` is the unique set to remember in the
    cache record. The remaining fields surface the private-only skip outcome so
    the orchestrator can log it, name it in the summary, and decide whether to
    cache the title as done.
    """

    seadex_dict: SeadexDict
    # The hash-filter path appends every url's hash unconditionally, and a
    # private torrent has no infohash, so this can hold None entries (the
    # release-group path filters those out, but the type must cover both).
    torrent_hashes: list[str | None]
    private_only_skipped: bool = False
    private_only_groups: list[str] = field(default_factory=list[str])
    skip_notices: list[SkipNotice] = field(default_factory=list[SkipNotice])


def normalize_rg(name: str | None) -> str | None:
    """Normalize a release group name for comparison

    Delegates to :func:`~.manual_import.normalize_group` (strip whitespace and
    wrapping dashes, casefold) so the grab-time filter and the import-time
    never-overwrite check share ONE normalization; this wrapper only adds the
    None-tolerance. Returns None for a missing/blank name.

    Args:
        name (str | None): Release group name
    """

    if not name:
        return None
    return normalize_group(name)


def get_episode_keys(
    all_episodes: Iterable[EpisodeRecord],
) -> set[tuple[int | None, int | None]]:
    """Build the set of (season, episode) keys an episode list covers

    Reduces a release's parsed episode list to the set of (season, episode)
    pairs it contains, so different SeaDex release groups can be compared by
    what files they cover.

    Args:
        all_episodes (iterable): Parsed episode dicts with "season"/"episode"
    """

    return {(ep.season, ep.episode) for ep in all_episodes}


def get_same_files_groups(seadex_dict: SeadexDict) -> list[list[str]]:
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

    # The grouping key is one of three shapes: a shared "all cover one movie"
    # sentinel str, a per-group "couldn't parse" sentinel tuple, or the parsed
    # episode-coverage frozenset that equates groups covering identical files.
    grouped: dict[
        str | tuple[str, str] | frozenset[tuple[int | None, int | None]],
        list[str],
    ] = {}
    for rg, rg_item in seadex_dict.items():
        all_episodes = rg_item.all_episodes

        key: str | tuple[str, str] | frozenset[tuple[int | None, int | None]]
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


def _is_flagged(rg_item: SeadexReleaseGroupItem) -> bool:
    return any(u.download for u in rg_item.urls.values())


def _is_public_group(rg_item: SeadexReleaseGroupItem) -> bool:
    return any(u.is_public for u in rg_item.urls.values())


def _is_fallback_group(rg_item: SeadexReleaseGroupItem) -> bool:
    return any(u.is_fallback for u in rg_item.urls.values())


def _all_public(rg_item: SeadexReleaseGroupItem) -> bool:
    return all(u.is_public for u in rg_item.urls.values())


def _flagged_all_addable(rg_item: SeadexReleaseGroupItem) -> bool:
    # No flagged private url (the add-time gate refuses those).
    return all(u.is_public for u in rg_item.urls.values() if u.download)


def _unflag(rg_item: SeadexReleaseGroupItem, dropped: list[SeadexUrlItem]) -> None:
    """Unflag a whole group, recording the urls this pass actually dropped."""

    for u in rg_item.urls.values():
        if u.download:
            dropped.append(u)
        u.download = False


def get_all_seadex_rgs_per_episode(
    seadex_dict: SeadexDict,
    sonarr_by_key: dict[tuple[int, int], SonarrEpisode],
) -> dict[str, set[str | None]]:
    """Get a list of all SeaDex releases per-episode

    Args:
        seadex_dict: Dictionary of SeaDex releases
        sonarr_by_key: Sonarr episodes indexed by (season, episode). A parsed
            SeaDex (season, episode) is recorded only when Sonarr has it, which
            this makes an O(1) key lookup. Built once by the caller and shared
            with the per-episode match loop in filter_by_release_group.
    """

    all_seadex_rgs_per_episode: dict[str, set[str | None]] = {"all": set()}

    if len(seadex_dict) > 1:
        for seadex_rg, seadex_rg_item in seadex_dict.items():
            # Index by the normalized name so the membership checks in
            # filter_by_release_group are case- and dash-insensitive
            seadex_rg_normalized = normalize_rg(seadex_rg)

            seadex_urls = seadex_rg_item.urls
            for url_item in seadex_urls.values():
                seadex_episodes = url_item.episodes

                # If we haven't managed to parse, then set this up as an
                # "all" episode fallback
                if len(seadex_episodes) == 0:
                    all_seadex_rgs_per_episode["all"].add(seadex_rg_normalized)

                for seadex_ep in seadex_episodes:
                    season = seadex_ep.season
                    episode = seadex_ep.episode

                    # Only record episodes Sonarr actually has, matching the
                    # original per-episode gate against the episode list
                    if (season, episode) in sonarr_by_key:
                        season_key = f"S{season:02d}E{episode:02d}"
                        all_seadex_rgs_per_episode.setdefault(
                            season_key,
                            set(),
                        ).add(seadex_rg_normalized)

    return all_seadex_rgs_per_episode


class DownloadPlanner:
    """Decides which SeaDex releases to grab for one AniList entry.

    Constructed once per run with the two config flags it consults; every
    decision method takes the already-shaped ``seadex_dict`` plus the Arr's
    release info as arguments and returns a :class:`PlanResult`. The planner
    keeps a logger only for the per-release debug breadcrumbs; the user-facing
    private-only skip is returned as a :class:`SkipNotice`, never logged here.
    """

    def __init__(
        self,
        *,
        interactive: bool,
        use_torrent_hash_to_filter: bool,
        logger: logging.Logger,
    ) -> None:
        self.interactive = interactive
        self.use_torrent_hash_to_filter = use_torrent_hash_to_filter
        self.logger = logger

    def plan(
        self,
        *,
        seadex_dict: SeadexDict,
        arr: Arr,
        arr_release_dict: ArrReleaseDict,
        cached_hashes: list[str | None],
        ep_list: list[SonarrEpisode] | None = None,
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
        seadex_dict: SeadexDict,
        cached_hashes: list[str | None],
    ) -> PlanResult:
        """Select downloads if the torrent hash is not already in the cache

        Multiple "best" releases are all grabbed, except where several cover
        the same files (see reduce_overlapping_downloads), in which case only
        one is kept

        Args:
            seadex_dict: Dictionary of SeaDex releases
            cached_hashes: Torrent hashes already remembered for this entry
        """

        torrent_hashes: list[str | None] = []

        for seadex_rg, seadex_rg_item in seadex_dict.items():
            self.logger.debug(
                indent_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.urls
            for url_item in seadex_urls.values():
                infohash = url_item.infohash

                # Dedup by infohash. KNOWN LIMITATION of this opt-in mode: a hashless
                # (private) release has infohash=None and the cache keeps a single None
                # marker, so a 2nd DISTINCT hashless release for an entry collapses to it
                # and is skipped (the first run still grabs all hashless releases present).
                torrent_hashes.append(infohash)
                if infohash not in cached_hashes:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {infohash} not found in cache. Will add to downloads",
                        ),
                    )

                    url_item.download = True

                elif infohash is None:
                    self.logger.debug(
                        indent_string(
                            "Hashless release already represented by the cache's None marker; skipping (see above)",
                        ),
                    )

                else:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {infohash} in cache. Will skip download",
                        ),
                    )

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring a public group)
        skips = self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            private_only_skipped=skips.skipped,
            private_only_groups=skips.groups,
            skip_notices=skips.notices,
        )

    def filter_by_release_group(
        self,
        seadex_dict: SeadexDict,
        arr: Arr,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
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
        # fresh scan of the whole list. The first entry wins on a duplicate key
        # (Sonarr episodes are unique by season+episode).
        sonarr_by_key: dict[tuple[int, int], SonarrEpisode] = {}
        for sonarr_ep in ep_list or []:
            sonarr_by_key.setdefault(
                season_episode_key(sonarr_ep.season_number, sonarr_ep.episode_number),
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

            seadex_urls = seadex_rg_item.urls
            for url, url_item in seadex_urls.items():
                seadex_episodes = url_item.episodes

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
        # Arr has none of them, only grab one (preferring a public group)
        skips = self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        # Build the hash list from whatever is still flagged for download, so it
        # always matches the exact set of torrents we'll add. Private torrents
        # have no infohash, so skip those
        torrent_hashes: list[str | None] = [
            url_item.infohash
            for rg_item in seadex_dict.values()
            for url_item in rg_item.urls.values()
            if url_item.download and url_item.infohash is not None
        ]

        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            private_only_skipped=skips.skipped,
            private_only_groups=skips.groups,
            skip_notices=skips.notices,
        )

    def _match_url_no_episodes(
        self,
        *,
        seadex_rg: str,
        url: str,
        url_item: SeadexUrlItem,
        arr: Arr,
        arr_release_dict: ArrReleaseDict,
        arr_release_groups: Iterable[str | None],
        overlapping_results: bool,
    ) -> None:
        """Decide a single url with no parsed episodes, by release group + size.

        Flips ``url_item.download`` in place. The blunt fallback used for
        Radarr and weirdly named TV: if the group isn't in the Arr's releases
        (and nothing overlaps) grab it; if it is, grab it only when the file
        sizes are disjoint.
        """

        if seadex_rg in arr_release_groups:
            # The group matches: fall through to a size comparison.
            seadex_file_sizes = url_item.size
            arr_file_sizes = as_size_list(arr_release_dict[seadex_rg])

            # If we have no overlaps at all, then add
            if set(seadex_file_sizes).isdisjoint(arr_file_sizes):
                self.logger.debug(
                    indent_string(
                        f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                        f"{', '.join([str(x) for x in arr_release_groups])}, but file sizes do not match - will download {url}",
                    ),
                )

                url_item.download = True
                url_item.size_mismatch = True

            else:
                self.logger.debug(
                    indent_string(
                        f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                        f"{', '.join([str(x) for x in arr_release_groups])}, and file sizes match",
                    ),
                )
        elif not overlapping_results:
            self.logger.debug(
                indent_string(
                    f"SeaDex release group {seadex_rg} not in {arr.capitalize()} releases: "
                    f"{', '.join([str(x) for x in arr_release_groups])} - will download {url}",
                ),
            )

            url_item.download = True
        else:
            # Group absent, but the Arr already holds another SeaDex-preferred
            # group's release covering these files - nothing to flag.
            self.logger.debug(
                indent_string(
                    f"SeaDex release group {seadex_rg} not in {arr.capitalize()} releases, but another "
                    f"SeaDex group already overlaps them - not flagging {url}",
                ),
            )

    def _match_url_episodes(
        self,
        *,
        seadex_rg: str,
        url: str,
        url_item: SeadexUrlItem,
        arr: Arr,
        seadex_episodes: list[EpisodeRecord],
        sonarr_by_key: dict[tuple[int, int], SonarrEpisode],
        all_seadex_rgs_per_episode: dict[str, set[str | None]],
        has_ep_list: bool,
        debug_on: bool,
    ) -> None:
        """Decide a single url against its parsed episodes, per episode.

        Flips ``url_item.download`` in place. For each parsed SeaDex episode
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
            seadex_ep_season = seadex_ep.season
            seadex_ep_episode = seadex_ep.episode
            seadex_ep_size = seadex_ep.size

            # A parsed episode with no season/episode can't key into the Sonarr
            # index (its keys are always concrete ints), and the SxxExx label
            # below needs both anyway, so skip it.
            if seadex_ep_season is None or seadex_ep_episode is None:
                continue

            # O(1) lookup into the indexed Sonarr episodes instead of
            # re-scanning the whole list for every parsed episode
            sonarr_ep = sonarr_by_key.get(
                (seadex_ep_season, seadex_ep_episode),
            )
            if sonarr_ep is None:
                continue

            # Get the matched Sonarr episode's file size
            sonarr_ep_size = sonarr_ep.episode_file.size if sonarr_ep.episode_file else None

            # Do the sizes match? A missing Sonarr file reports no
            # size, so guard against None == None reading as a match
            # when neither side actually has a size.
            size_match = sonarr_ep_size is not None and sonarr_ep_size == seadex_ep_size

            season_ep_str = f"S{seadex_ep_season:02d}E{seadex_ep_episode:02d}"

            # Check SeaDex release group matches the episode release group in Sonarr
            sonarr_rg = sonarr_ep.episode_file.release_group if sonarr_ep.episode_file else None
            sonarr_rg_normalized = normalize_rg(sonarr_rg)
            seadex_rg_normalized = normalize_rg(seadex_rg)
            # If not, flag as should be downloaded if it's not
            # already in some overlapping release.
            # normalized name indexes all_seadex_rgs_per_episode, so compare the normalized name
            if (
                sonarr_rg_normalized != seadex_rg_normalized
                and sonarr_rg_normalized not in all_seadex_rgs_per_episode["all"]
            ):
                # Avoid duplicating when another release already covers it
                all_seadex_rg = all_seadex_rgs_per_episode.get(
                    season_ep_str,
                    (),
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

                    url_item.download = True

            else:
                if debug_on:
                    self.logger.debug(
                        indent_string(
                            f"Found SeaDex match to {arr.capitalize()} for {season_ep_str}.",
                        ),
                    )
                    if not size_match:
                        self.logger.debug(
                            indent_string(
                                f"-> Sizes are different: {sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
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
            url_item.download = True
            url_item.size_mismatch = True

    def reduce_overlapping_downloads(
        self,
        seadex_dict: SeadexDict,
    ) -> PrivateOnlySkips:
        """Reduce overlapping flagged downloads down to a single release group

        Where multiple preferred release groups cover the same files and the
        Arr doesn't already have any of them, we only want to grab one. We
        prefer a public release group and drop the private ones (private
        releases are never grabbed). If the only options are private, we
        record a warning SkipNotice and skip the title (without caching it as
        done) rather than grabbing a private release - unless an unflagged
        public group covering the same files rides along (a
        ``private_releases: fallback`` stand-in or a preferred public pick).
        Then the private groups are dropped with an INFO notice instead: the
        public group is promoted (grabbed) when a private flag was a
        size-mismatch upgrade, and left alone when the Arr genuinely already
        owns its files.

        After each set resolves, just-dropped public urls whose coverage no
        surviving url carries are re-flagged (group-atomic drops must not lose
        episodes). Finally, within each group, flagged urls carrying identical
        non-empty file-name sets (cross-seeded copies of one release) are
        deduped to the first.

        Mutates the download flags on seadex_dict in place and returns the
        private-only skip outcome (skipped flag, group names, notices to log).
        Skipped entirely in interactive mode, where the user has already
        hand-picked what to grab.

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        skips = PrivateOnlySkips()

        # In interactive mode the user has explicitly chosen which releases to
        # grab, so don't second-guess them by dropping any
        if self.interactive:
            return skips

        for same_files in get_same_files_groups(seadex_dict):
            # Only the release groups the Arr doesn't already have are flagged
            flagged = [rg for rg in same_files if _is_flagged(seadex_dict[rg])]
            if len(flagged) == 0:
                continue

            dropped = self._reduce_same_files_set(seadex_dict, same_files, flagged, skips)
            self._rescue_dropped_coverage(seadex_dict, same_files, dropped)

        # Within ONE group, flagged urls with identical non-empty file-name sets
        # are the same release cross-seeded (distinct infohashes = duplicate
        # downloads; the promotion branch above can flip several at once): keep
        # the first, unflag the rest. An empty fileset can't prove identity, so
        # it's never deduped; cross-group overlap is the same-files logic above.
        for rg_item in seadex_dict.values():
            seen: set[frozenset[str]] = set()
            for u in rg_item.urls.values():
                if not u.download or not u.files:
                    continue
                file_names = frozenset(u.files)
                if file_names in seen:
                    u.download = False
                else:
                    seen.add(file_names)

        return skips

    def _reduce_same_files_set(
        self,
        seadex_dict: SeadexDict,
        same_files: list[str],
        flagged: list[str],
        skips: PrivateOnlySkips,
    ) -> list[SeadexUrlItem]:
        """Resolve ONE same-files set down to a single keeper (or a skip).

        Appends any notices/skip state to ``skips`` and returns the urls this
        pass unflagged, for the coverage rescue.
        """

        dropped: list[SeadexUrlItem] = []

        public_flagged = [rg for rg in flagged if _is_public_group(seadex_dict[rg])]
        # The held-stale-not-owned marker: a size-mismatch flag means the Arr
        # holds the release at a STALE size (upgrade pending), so an unflagged
        # public group in this set is NOT owned and may be promoted in its
        # place. Without it, promotion would re-download owned content.
        upgrade_pending = any(u.size_mismatch for rg in flagged for u in seadex_dict[rg].urls.values())

        if len(public_flagged) == 0:
            # An unflagged public group in THIS same-files set can stand in
            # (fallback or preferred; a flagged one would have taken the keeper
            # branch below). A public group covering OTHER files doesn't excuse
            # dropping this set, so the gate is per-set, never per-entry.
            if upgrade_pending:
                promoted = self._promote_public_alternative(seadex_dict, same_files)
                if promoted is not None:
                    self._drop_promoted_over(seadex_dict, flagged, "private-only", promoted, skips, dropped)
                    return dropped
            elif any(_is_fallback_group(seadex_dict[rg]) for rg in same_files):
                # Unflagged with no size mismatch: the Arr genuinely already
                # owns the fallback's files.
                skips.notices.append(
                    SkipNotice(
                        groups=list(flagged),
                        reason="private-only; a public fallback already covers these files",
                        level=logging.INFO,
                    ),
                )
                for rg in flagged:
                    _unflag(seadex_dict[rg], dropped)
                return dropped

            # The Arr has none of these release groups, private grabs are off,
            # and no public url covers the same files. Don't grab a private
            # release, just record a skip notice and skip. Flag the skip so the
            # caller doesn't cache the title as done.
            skips.notices.append(
                SkipNotice(
                    groups=list(flagged),
                    reason="private-only (private releases not allowed)",
                    level=logging.WARNING,
                ),
            )
            skips.skipped = True
            skips.groups.extend(flagged)
            for rg in flagged:
                _unflag(seadex_dict[rg], dropped)
            return dropped

        # Keep the first public release group whose flagged urls are all
        # addable: a mixed group's flagged private url is refused at add time,
        # losing the coverage only it carries, so a fully-addable group wins
        # over it.
        keeper = next((rg for rg in public_flagged if _flagged_all_addable(seadex_dict[rg])), None)
        if keeper is None:
            # No fully-addable group: when the private flags are stale-size
            # upgrades, an unflagged public group covering this set still grabs
            # cleanly - promote it rather than degrading to an add-time refusal.
            if upgrade_pending:
                promoted = self._promote_public_alternative(seadex_dict, same_files)
                if promoted is not None:
                    self._drop_promoted_over(
                        seadex_dict,
                        flagged,
                        "remaining files private-only",
                        promoted,
                        skips,
                        dropped,
                    )
                    return dropped
            # Degrade to the first public group (the add-time gate then warns).
            keeper = public_flagged[0]

        # Tell the user when the keeper is a non-preferred fallback standing in
        # for private preferred picks (a plain public-over-private keeper stays
        # a debug line, as ever).
        private_dropped = [rg for rg in flagged if not _is_public_group(seadex_dict[rg])]
        if private_dropped and _is_fallback_group(seadex_dict[keeper]):
            skips.notices.append(
                SkipNotice(
                    groups=private_dropped,
                    reason=f"private-only; falling back to {keeper}",
                    level=logging.INFO,
                ),
            )

        self._drop_losers(seadex_dict, flagged, keeper, dropped)
        return dropped

    def _drop_losers(
        self,
        seadex_dict: SeadexDict,
        flagged: list[str],
        keeper: str,
        dropped: list[SeadexUrlItem],
    ) -> None:
        """Unflag every flagged group but the keeper, recording the drops."""

        for rg in flagged:
            if rg == keeper:
                continue

            self.logger.debug(
                indent_string(
                    f"Not downloading release group {rg}: release group {keeper} already covers the same files",
                ),
            )
            _unflag(seadex_dict[rg], dropped)

    @staticmethod
    def _promote_public_alternative(seadex_dict: SeadexDict, same_files: list[str]) -> str | None:
        """Flip on an unflagged public group covering this set; None if there is none.

        Prefer a fully-public group - promotion only flips public urls, so a
        mixed group's private url would leave part of the set's coverage
        ungrabbed.
        """

        candidates = [rg for rg in same_files if not _is_flagged(seadex_dict[rg]) and _is_public_group(seadex_dict[rg])]
        if not candidates:
            return None

        promoted = next((rg for rg in candidates if _all_public(seadex_dict[rg])), candidates[0])
        for u in seadex_dict[promoted].urls.values():
            if u.is_public:
                u.download = True
        return promoted

    @staticmethod
    def _drop_promoted_over(
        seadex_dict: SeadexDict,
        flagged: list[str],
        prefix: str,
        promoted: str,
        skips: PrivateOnlySkips,
        dropped: list[SeadexUrlItem],
    ) -> None:
        """Drop the flagged groups a promoted public group now stands in for.

        The INFO notice says "falling back to" only for a genuine fallback
        stand-in; a promoted preferred group reads "grabbing public alternative".
        """

        verb = "falling back to" if _is_fallback_group(seadex_dict[promoted]) else "grabbing public alternative"
        skips.notices.append(
            SkipNotice(
                groups=list(flagged),
                reason=f"{prefix}; {verb} {promoted}",
                level=logging.INFO,
            ),
        )
        for rg in flagged:
            _unflag(seadex_dict[rg], dropped)

    def _rescue_dropped_coverage(
        self,
        seadex_dict: SeadexDict,
        same_files: list[str],
        dropped: list[SeadexUrlItem],
    ) -> None:
        """Re-flag just-dropped public urls whose coverage no survivor carries.

        A group-atomic drop can lose episodes: two mixed groups with equal
        coverage unions land in one set, and unflagging the loser wholesale
        drops its public url's unique coverage. Those urls were flagged by the
        matcher (the Arr provably lacks their files) before this pass dropped
        them, so re-flagging can't re-download owned content. Scoped to THIS
        set's just-dropped urls only - never matcher-unflagged ones - and a url
        with no parsed episodes (movies, unparsed groups) has no coverage
        vocabulary, so it is never rescued.
        """

        rescuable = [u for u in dropped if u.is_public and u.episodes]
        if not rescuable:
            return

        # What the set will actually obtain: surviving flagged urls the add-time
        # gate accepts (private urls are refused there, so they never count).
        survivor_keys = {
            key
            for rg in same_files
            for u in seadex_dict[rg].urls.values()
            if u.download and u.is_public
            for key in get_episode_keys(u.episodes)
        }
        for u in rescuable:
            url_keys = get_episode_keys(u.episodes)
            if url_keys <= survivor_keys:
                continue
            self.logger.debug(
                indent_string(
                    f"Re-flagging {u.url}: no surviving release covers its episodes",
                ),
            )
            u.download = True
            survivor_keys |= url_keys

    @staticmethod
    def get_any_to_download(seadex_dict: SeadexDict) -> bool:
        """Check if any torrents are marked as to download

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        return any(url_item.download for rg_item in seadex_dict.values() for url_item in rg_item.urls.values())
