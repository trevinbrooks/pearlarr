"""The download-decision engine: which SeaDex releases to grab.

`DownloadPlanner` is near-pure: it consumes the shaped `seadex_dict`, the
Arr's current release info, an optional episode list, and the cached torrent
hashes, and returns a `PlanResult`. It flips the per-url `download`
flags in place and reports the private-only skip outcome (`PlanResult.skips`:
*what to log* and *what was skipped*) as data, rather than reaching into the
orchestrator's run state or its log formatter.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import compress
from typing import NamedTuple

from .config import Arr
from .manual_import import GuardFacts, OwnedEpisode, normalize_rg
from .output import Severity
from .seadex_types import (
    ArrReleaseDict,
    EpisodeKey,
    EpisodeRecord,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
    as_size_list,
    index_episodes_by_key,
    season_episode_key,
)


@dataclass
class SkipNotice:
    """A release dropped for being private, for the caller to log.

    Rendered by the orchestrator as `"<groups> <reason>"` on a `skipped`
    detail line. A warn-and-hold skip logs at WARNING. A drop covered by a
    public fallback logs at INFO.
    """

    groups: list[str]
    reason: str
    severity: Severity = Severity.WARNING


@dataclass
class PrivateOnlySkips:
    """The private-only skip outcome of `reduce_overlapping_downloads`."""

    skipped: bool = False
    """True when at least one set of same-files release groups was dropped because none were available
    publicly and no public fallback covered the same files."""
    groups: list[str] = field(default_factory=list[str])
    """The dropped groups' names, for the run summary."""
    notices: list[SkipNotice] = field(default_factory=list[SkipNotice])
    """What to log for the drop."""
    stale_held: bool = False
    """A hold where the Arr owns the preferred private release at a stale size and only a fallback could
    stand in. A fallback never replaces an owned copy."""
    fallback_covered: bool = False
    """The owned-fallback soft-skip: the Arr genuinely owns a public fallback's files. Drives the cache's
    fallback-satisfied marker."""


@dataclass
class PlanResult:
    """The download-decision engine's output.

    Carries the annotated dict, the hashes to remember, and the private-only skip
    outcome (`skips`) so the orchestrator can log it, name it in the summary, and
    decide whether to cache the title as done.
    """

    seadex_dict: SeadexDict
    """The same dict passed in, annotated in place with per-url `download` flags."""
    torrent_hashes: list[str | None]
    """The unique torrent infohashes to remember. None for a hashless private torrent."""
    skips: PrivateOnlySkips
    """The private-only skip outcome from `reduce_overlapping_downloads`, folded onto run state by the caller."""
    guards: GuardFacts = field(default_factory=GuardFacts)
    """The overwrite-guard evidence the import inherits whole (see `GuardFacts`):
    the size-verified current groups, the positively-stale ones, and the owned
    untagged `(episode id, size)` claims. All-empty for Radarr and the hash path,
    which gather no size evidence and so protect only grabbed groups."""


def _render_groups(groups: Iterable[str | None]) -> str:
    """Comma-join release group names for a log line, rendering None as `(none)`."""

    return ", ".join(rg or "(none)" for rg in groups)


def _untagged_counter(untagged_sizes: Iterable[int]) -> Counter[int] | None:
    """The untagged on-disk multiset, or None when it can never claim ownership.

    Folded once per entry (the ownership checks run per url). None when empty,
    or when any size is non-positive - a zero-byte failed copy proves nothing
    and stays grabbable, so the whole multiset is disqualified with it.
    """

    counter = Counter(untagged_sizes)
    if not counter or any(size <= 0 for size in counter):
        return None
    return counter


def _owns_untagged_copy(counter: Counter[int] | None, listed_sizes: Iterable[int]) -> bool:
    """Whether every file the arr holds untagged belongs to this listing, by exact size.

    Identifies an owned copy whose group tag a rename stripped. Containment runs
    arr -> listing, never the reverse: a listing also carries files the arr never
    holds (subtitles, fonts, creditless extras), so requiring every LISTED size
    on disk would only ever match a single-file release. Multisets, so two
    untagged files at one size need two listed twins. `counter` is the
    `_untagged_counter` fold (None short-circuits: nothing ownable on disk).
    """

    return counter is not None and counter <= Counter(listed_sizes)


def get_episode_keys(
    all_episodes: Iterable[EpisodeRecord],
) -> set[tuple[int | None, int | None]]:
    """Build the set of (season, episode) keys an episode list covers.

    Reduces a release's parsed episode list to the set of (season, episode)
    pairs it contains, so different SeaDex release groups can be compared by
    what files they cover.
    """

    return {(ep.season, ep.episode) for ep in all_episodes}


def get_same_files_groups(seadex_dict: SeadexDict) -> list[list[str]]:
    """Group SeaDex release groups that cover exactly the same files.

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
    """Unflag a whole group, recording the urls this pass actually dropped.

    `upgrade` clears with `download` - it describes the grab, so it must never
    outlive it (a rendered release line would otherwise mark a dropped url as
    an upgrade that no longer happens).
    """

    for u in rg_item.urls.values():
        if u.download:
            dropped.append(u)
        u.download = False
        u.upgrade = False


class EpisodeCoverage(NamedTuple):
    """Which sibling picks cover each episode - the matcher's duplicate-avoidance index.

    Consulted when an on-disk file's group mismatches the url being decided:
    a covering group means another recommended release already accounts for
    the episode, so no duplicate grab. Membership is by normalized name; a
    group whose name normalizes to None is indexed NOWHERE - an unidentifiable
    listing group must never read as covering an untagged on-disk file.
    """

    blanket: frozenset[str]
    """Groups with an unparsed url: coverage unprovable, so they cover every episode unconditionally."""

    by_key: Mapping[EpisodeKey, frozenset[str]]
    """The groups whose parsed urls cover each (season, episode) Sonarr actually has."""


def episode_coverage(
    seadex_dict: SeadexDict,
    sonarr_by_key: Mapping[EpisodeKey, SonarrEpisode],
) -> EpisodeCoverage:
    """Index which SeaDex groups cover each episode, by normalized name.

    `sonarr_by_key` gates recording to episodes Sonarr has (an O(1) key
    lookup). Built once by the caller and shared with the per-episode match
    loop in filter_by_release_group. A single-group entry skips the build:
    coverage only ever excuses a mismatch against a SIBLING pick.
    """

    if len(seadex_dict) < 2:
        return EpisodeCoverage(frozenset(), {})

    blanket: set[str] = set()
    by_key: dict[EpisodeKey, set[str]] = {}
    for seadex_rg, seadex_rg_item in seadex_dict.items():
        # Index by the normalized name so the membership checks in
        # filter_by_release_group are case- and dash-insensitive.
        seadex_rg_normalized = normalize_rg(seadex_rg)
        if seadex_rg_normalized is None:
            continue

        for url_item in seadex_rg_item.urls.values():
            seadex_episodes = url_item.episodes

            # An unparsed url proves nothing about coverage, so its group
            # covers everything rather than nothing.
            if len(seadex_episodes) == 0:
                blanket.add(seadex_rg_normalized)

            for seadex_ep in seadex_episodes:
                season = seadex_ep.season
                episode = seadex_ep.episode
                if season is None or episode is None:
                    continue
                key = EpisodeKey(season, episode)
                if key in sonarr_by_key:
                    by_key.setdefault(key, set()).add(seadex_rg_normalized)

    return EpisodeCoverage(frozenset(blanket), {key: frozenset(groups) for key, groups in by_key.items()})


class _EpisodeIdentity(NamedTuple):
    """The effective release identity of one on-disk episode file."""

    group: str | None
    """The normalized group: the file's own tag, or the pick a listed size named."""
    by_size: bool
    """Identified by exact listed size, not a tag (see `_episode_identities`)."""
    size: int | None
    """The file's on-disk size (always the listed size for a `by_size` identity)."""


class _EntryIdentities(NamedTuple):
    """`_episode_identities`' two views of the on-disk files - deliberately OVERLAPPING.

    `identities` holds every identified file (tagged or size-identified);
    `untagged_by_key` holds every untagged file, the size-identified ones
    included, because the unparseable-url matcher compares that multiset
    whole. The overlap is the size-identified files - do not join the views.
    """

    identities: dict[EpisodeKey, _EpisodeIdentity]
    """Each identified file's effective release identity."""

    untagged_by_key: dict[EpisodeKey, int]
    """Every untagged on-disk file's size (zero/unknown folded to 0)."""


def _episode_identities(
    seadex_dict: SeadexDict,
    sonarr_by_key: Mapping[EpisodeKey, SonarrEpisode],
) -> _EntryIdentities:
    """Resolve each on-disk file's effective identity: its group tag, or the pick its size names.

    A library rename scheme can strip the release-group tag off a file we
    already hold, leaving it unidentifiable by name. A POSITIVE exact size match
    against a pick's listed episode still says whose copy it is (a zero or
    unknown size proves nothing, so a failed copy stays grabbable).

    Entry-wide on purpose: ownership is a property of the FILE, so every pick of
    the entry must read it the same way. Deciding it per-url instead would let
    the pick that matched skip its download while a sibling pick still grabbed
    the same episodes. The first pick to claim an episode wins - a tie means two
    picks list it at the same size, where either answer reads as owned.
    """

    identities: dict[EpisodeKey, _EpisodeIdentity] = {}
    # Every untagged on-disk file. A zero/unknown size folds to 0: it can never
    # size-match an identity below, and in the whole-copy multiset it vetoes
    # ownership (a failed copy proves nothing and stays grabbable). Almost
    # always empty (a tagged library), which skips the listing scan.
    untagged: dict[EpisodeKey, int] = {}
    for key, sonarr_ep in sonarr_by_key.items():
        arr_file = sonarr_ep.episode_file
        if arr_file is None:
            continue
        group = normalize_rg(arr_file.release_group)
        if group is not None:
            identities[key] = _EpisodeIdentity(group, by_size=False, size=arr_file.size)
        else:
            untagged[key] = arr_file.size or 0
    if not untagged:
        return _EntryIdentities(identities, untagged)

    for seadex_rg, rg_item in seadex_dict.items():
        normalized = normalize_rg(seadex_rg)
        if normalized is None:
            continue
        for url_item in rg_item.urls.values():
            for seadex_ep in url_item.episodes:
                if seadex_ep.season is None or seadex_ep.episode is None:
                    continue
                key = season_episode_key(seadex_ep.season, seadex_ep.episode)
                if key in identities or seadex_ep.size <= 0 or untagged.get(key) != seadex_ep.size:
                    continue
                identities[key] = _EpisodeIdentity(normalized, by_size=True, size=seadex_ep.size)
    return _EntryIdentities(identities, untagged)


@dataclass(frozen=True, slots=True)
class _MatchContext:
    """Per-entry invariants for the URL match loop.

    Computed once per entry in `filter_by_release_group` and shared by both
    per-URL matchers. Nothing here changes inside the loop (only each
    `url_item`'s `download`/`upgrade` fields flip).
    """

    arr_release_dict: ArrReleaseDict
    arr_sizes_by_norm: Mapping[str | None, list[int]]
    """The Arr's file sizes merged under normalized group names. Built once per entry rather than per URL."""
    overlapping_results: bool
    """Some pick's release is already on disk: by group tag, or as the untagged
    copy a pick's listed sizes identified whole (so a sibling pick is never
    grabbed over a copy only a rename left unrecognizable)."""
    sonarr_by_key: dict[EpisodeKey, SonarrEpisode]
    episode_identities: Mapping[EpisodeKey, _EpisodeIdentity]
    """Each on-disk file's effective release identity (see `_episode_identities`)."""
    untagged_counter: Counter[int] | None
    """The untagged on-disk files' size multiset (Radarr's `None` key, or Sonarr's
    untagged episode files), pre-folded by `_untagged_counter` - the unparseable-url
    ownership input. None when nothing untagged can claim ownership."""
    coverage: EpisodeCoverage
    """Which sibling picks cover each episode (see `episode_coverage`)."""
    has_ep_list: bool
    debug_on: bool


class _GroupVerdicts(NamedTuple):
    """The plan's size-evidence verdicts per pick group, feeding `GuardFacts` by name."""

    current: tuple[str, ...]
    """Groups whose every attributed on-disk file sits at a listed size (vacuously, none on disk)."""

    stale: tuple[str, ...]
    """Groups with an attributed file at a size no sized listing carries."""


def _group_verdicts(seadex_dict: SeadexDict, ctx: _MatchContext) -> _GroupVerdicts:
    """Judge each pick group's on-disk copy by size evidence: `(current, stale)` names.

    Disk-centric: a group is CURRENT only when every on-disk file attributed to
    it (by tag, or by the size identification) sits at a size some url of the
    group lists - vacuously when it has nothing on disk. Any attributed file at
    a size no sized listing carries makes it STALE. A listing with no sizes is
    no evidence either way, so an all-blind group with files on disk lands in
    NEITHER list: unprotected at import, but never voted an upgrade. Whole-group
    verdicts are conservative in the overwrite direction - one stale file exits
    the group from the never-overwrite set entirely.
    """

    cover: dict[str | None, set[int]] = {}
    for rg, item in seadex_dict.items():
        listed = cover.setdefault(normalize_rg(rg), set())
        for url_item in item.urls.values():
            # Url-level and per-episode sizes both count (an episode parse can
            # carry sizes a bare listing record doesn't); zero proves nothing.
            listed.update(s for s in url_item.size if s > 0)
            listed.update(ep.size for ep in url_item.episodes if ep.size > 0)

    # The on-disk files each group must answer for. With an episode list the
    # per-episode identities attribute them (size-identified files included);
    # without one, the Arr's tag-keyed size map is the whole vocabulary.
    attributed: dict[str | None, list[int | None]] = {}
    if ctx.has_ep_list:
        for identity in ctx.episode_identities.values():
            attributed.setdefault(identity.group, []).append(identity.size)
    else:
        attributed = {norm: list(sizes) for norm, sizes in ctx.arr_sizes_by_norm.items() if norm is not None}

    current: list[str] = []
    stale: list[str] = []
    for rg in seadex_dict:
        norm = normalize_rg(rg)
        files = attributed.get(norm, [])
        listed = cover.get(norm, set())
        if not files:
            current.append(rg)
        elif not listed:
            continue
        elif all(size is not None and size in listed for size in files):
            current.append(rg)
        elif any(size is not None and size not in listed for size in files):
            stale.append(rg)
        # A group whose only disagreements are unknown sizes stays unverifiable.
    return _GroupVerdicts(tuple(current), tuple(stale))


class DownloadPlanner:
    """Decides which SeaDex releases to grab for one AniList entry.

    Constructed once per Arr run with the arr it plans for and the two config
    flags it consults. Every decision method takes the already-shaped
    `seadex_dict` plus the Arr's release info as arguments and returns a
    `PlanResult`. The planner keeps a logger only for the per-release
    debug breadcrumbs. The operator-facing private-only skip is returned as a
    `SkipNotice`, never logged here.
    """

    def __init__(
        self,
        *,
        arr: Arr,
        interactive: bool,
        use_torrent_hash_to_filter: bool,
        logger: logging.Logger,
    ) -> None:
        self.arr = arr
        self.interactive = interactive
        self.use_torrent_hash_to_filter = use_torrent_hash_to_filter
        self.logger = logger

    def plan(
        self,
        *,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        cached_hashes: list[str | None],
        ep_list: list[SonarrEpisode] | None = None,
    ) -> PlanResult:
        """Flip the download flags and return the full plan for an entry.

        Selects the hash-based or release-group-based strategy from the config
        flag, then unions in the cached hashes (release-group path only - the
        hash path already lists every url's hash) and de-duplicates.
        `seadex_dict` is annotated in place. `cached_hashes` are the torrent
        hashes already remembered for this entry.
        """

        if self.use_torrent_hash_to_filter:
            result = self.filter_by_torrent_hash(
                seadex_dict=seadex_dict,
                cached_hashes=cached_hashes,
            )
        else:
            result = self.filter_by_release_group(
                seadex_dict=seadex_dict,
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
        """Select downloads if the torrent hash is not already in the cache.

        Multiple "best" releases are all grabbed, except where several cover
        the same files (see reduce_overlapping_downloads), in which case only
        one is kept
        """

        torrent_hashes: list[str | None] = []

        for seadex_rg, seadex_rg_item in seadex_dict.items():
            self.logger.debug(f"Filtering for release group {seadex_rg}")

            seadex_urls = seadex_rg_item.urls
            for url_item in seadex_urls.values():
                infohash = url_item.infohash

                # Dedup by infohash. KNOWN LIMITATION of this opt-in mode: a hashless
                # (private) release has infohash=None and the cache keeps a single None
                # marker, so a 2nd DISTINCT hashless release for an entry collapses to it
                # and is skipped (the first run still grabs all hashless releases present).
                torrent_hashes.append(infohash)
                if infohash not in cached_hashes:
                    self.logger.debug(f"Torrent hash {infohash} not found in cache. Will add to downloads")

                    url_item.download = True

                elif infohash is None:
                    self.logger.debug(
                        "Hashless release already represented by the cache's None marker; "
                        "skipping (hashless releases can't be told apart)",
                    )

                else:
                    self.logger.debug(f"Torrent hash {infohash} in cache. Will skip download")

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring a public group)
        skips = self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            skips=skips,
        )

    def filter_by_release_group(
        self,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> PlanResult:
        """Filter torrents by release group.

        This is either an episode-by-episode for the Sonarr
        case where we can parse episodes, or a more blunt
        hammer just checking against anything for Radarr
        and weirdly named TV
        """

        # The release-group names, used both for display (insertion order
        # preserved) and for membership tests below. A dict keys view already
        # supports `in` in O(1), so there's no need to materialize a list.
        arr_release_groups = arr_release_dict.keys()

        # And also just check if any release group matches any Arr release tag -
        # by normalized name, matching the per-episode path's comparison.
        seadex_keys = {normalize_rg(rg) for rg in seadex_dict}
        overlapping_results = any(normalize_rg(rg) in seadex_keys for rg in arr_release_groups)

        # Index the Sonarr episodes by (season, episode) once, shared by both
        # the overlap map below and the per-episode match loop: looking up a
        # parsed SeaDex (season, episode) is then an O(1) dict op rather than a
        # fresh scan of the whole list.
        sonarr_by_key = index_episodes_by_key(ep_list or [])

        # Which sibling picks cover each episode, reusing the index above
        coverage = episode_coverage(seadex_dict, sonarr_by_key)

        # Each on-disk file's effective identity - its tag, or the pick whose
        # listed size names an untagged file - decided once for the whole entry
        # so every pick's url reads the same ownership.
        identities, untagged_by_key = _episode_identities(seadex_dict, sonarr_by_key)

        # Resolve once: the per-episode debug lines below sit in the hot
        # matching loop, so this lets us skip building their f-strings on a
        # normal INFO run instead of formatting them only to discard them.
        debug_on = self.logger.isEnabledFor(logging.DEBUG)
        if debug_on:
            for (season, episode), identity in identities.items():
                if identity.by_size:
                    self.logger.debug(
                        f"{self.arr.capitalize()} file for S{season:02d}E{episode:02d} has no release group "
                        f"but matches {identity.group}'s listed size exactly - reading it as that release",
                    )

        # The Arr's sizes merged under normalized group names (like the
        # per-episode path's comparison). Loop-invariant - arr_release_dict is
        # never mutated below - so build it once per entry, not per URL.
        arr_sizes_by_norm: dict[str | None, list[int]] = {}
        for arr_rg, sizes in arr_release_dict.items():
            arr_sizes_by_norm.setdefault(normalize_rg(arr_rg), []).extend(as_size_list(sizes))

        # Every untagged file on disk, folded to its ownership multiset once per
        # entry: Radarr's None key, or the Sonarr episode files the identity
        # pass collected. A pick whose listed sizes contain the whole multiset
        # owns the copy, which counts as an on-disk overlap - a sibling pick
        # must hold exactly as it would had the copy kept its tag. None (the
        # common tagged library) skips the per-url compares entirely.
        untagged_counter = _untagged_counter((*arr_sizes_by_norm.get(None, []), *untagged_by_key.values()))
        overlapping_results = overlapping_results or (
            untagged_counter is not None
            and any(
                _owns_untagged_copy(untagged_counter, url_item.size)
                for rg_item in seadex_dict.values()
                for url_item in rg_item.urls.values()
            )
        )

        ctx = _MatchContext(
            arr_release_dict=arr_release_dict,
            arr_sizes_by_norm=arr_sizes_by_norm,
            overlapping_results=overlapping_results,
            sonarr_by_key=sonarr_by_key,
            episode_identities=identities,
            untagged_counter=untagged_counter,
            coverage=coverage,
            has_ep_list=ep_list is not None,
            debug_on=debug_on,
        )

        for seadex_rg, seadex_rg_item in seadex_dict.items():
            self.logger.debug(f"Filtering for release group {seadex_rg}")

            for url_item in seadex_rg_item.urls.values():
                # Simple case, we have no episode mappings, so
                # just fall back to checking against release group
                if not url_item.episodes:
                    self._match_url_no_episodes(ctx, seadex_rg, url_item)
                    continue

                self._match_url_episodes(ctx, seadex_rg, url_item)

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

        verdicts = _group_verdicts(seadex_dict, ctx)
        return PlanResult(
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            skips=skips,
            guards=GuardFacts(
                entry_groups=verdicts.current,
                stale_groups=verdicts.stale,
                # Resolved to (episode id, size) pairs here so the import seeds
                # inherit the exact files the grab decision just called ours -
                # and can re-verify them by size before honoring the claim.
                owned_episodes=tuple(
                    OwnedEpisode(eid, identity.size)
                    for key, identity in identities.items()
                    if identity.by_size and identity.size and (eid := sonarr_by_key[key].id)
                ),
            ),
        )

    def _match_url_no_episodes(
        self,
        ctx: _MatchContext,
        seadex_rg: str,
        url_item: SeadexUrlItem,
    ) -> None:
        """Decide a single url with no parsed episodes, by release group + size.

        Flips `url_item.download` in place. The blunt fallback used for
        Radarr and weirdly named TV: if the group isn't in the Arr's releases
        (and nothing overlaps) grab it. If it is, grab it only when the file
        sizes are disjoint. A copy a rename left untagged is identified by size
        instead (`_owns_untagged_copy`).
        """

        url = url_item.url
        # The release-group names, for the debug lines (insertion order preserved).
        arr_release_groups = ctx.arr_release_dict.keys()

        # Match by the normalized name (like the per-episode path). The merged
        # size index was built once per entry (see _MatchContext).
        seadex_norm = normalize_rg(seadex_rg)
        if seadex_norm in ctx.arr_sizes_by_norm:
            # The group matches: fall through to a size comparison. A listing
            # with no sizes carries no evidence either way, so the held copy
            # stands (a vacuous disjoint must not read as an upgrade).
            seadex_file_sizes = url_item.size
            arr_file_sizes = ctx.arr_sizes_by_norm[seadex_norm]

            if seadex_file_sizes and set(seadex_file_sizes).isdisjoint(arr_file_sizes):
                if ctx.debug_on:
                    self.logger.debug(
                        f"SeaDex release group {seadex_rg} in {self.arr.capitalize()} releases: "
                        f"{_render_groups(arr_release_groups)}, but file sizes do not match - will download {url}",
                    )

                url_item.download = True
                url_item.upgrade = True

            elif ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} in {self.arr.capitalize()} releases: "
                    f"{_render_groups(arr_release_groups)}, and no listed size contradicts the on-disk copy",
                )
        elif _owns_untagged_copy(ctx.untagged_counter, url_item.size):
            # A rename stripped the group tag: the untagged files on disk
            # (Radarr's None key, or Sonarr's untagged episode files) all
            # belong to this listing by size, so the copy is ours.
            if ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases: "
                    f"{_render_groups(arr_release_groups)}, but the untagged files on disk are all "
                    f"this release's - not flagging {url}",
                )
        elif not ctx.overlapping_results:
            if ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases: "
                    f"{_render_groups(arr_release_groups)} - will download {url}",
                )

            url_item.download = True
        elif ctx.debug_on:
            # Group absent, but the Arr already holds another SeaDex-preferred
            # group's release covering these files - nothing to flag.
            self.logger.debug(
                f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases, but another "
                f"SeaDex group already overlaps them - not flagging {url}",
            )

    def _match_url_episodes(
        self,
        ctx: _MatchContext,
        seadex_rg: str,
        url_item: SeadexUrlItem,
    ) -> None:
        """Decide a single url against its parsed episodes, per episode.

        Flips `url_item.download` in place. For each parsed SeaDex episode
        we check whether it exists in the Sonarr index, whether the release
        group matches, and whether the file sizes match. A release-group
        mismatch with no covering alternative, or an all-sizes mismatch among
        the rg-matched episodes, flips download on. An untagged on-disk file
        the entry's sizes identified (`_episode_identities`) matches by that
        group instead of by its missing tag.
        """

        # At this point, we need an episode list from Sonarr. A non-None but
        # empty list still runs the (no-op) loop below. Only an absent list skips.
        if not ctx.has_ep_list:
            self.logger.debug(
                "Skipping per-episode check: no Sonarr episode list available",
            )
            return

        url = url_item.url
        seadex_episodes = url_item.episodes

        # For each episode we've parsed from the torrent, check if a) it exists in the Sonarr list, b) if
        # the release group matches, and c) if the file sizes match. If there's any mismatch between release
        # groups (and there are no alternatives), then flip download to True. If all the sizes mismatch,
        # flip download to true

        rg_matches = [False] * len(seadex_episodes)
        size_matches = [False] * len(seadex_episodes)
        # Fixed per call - normalize once, not once per parsed episode (only
        # sonarr_rg_normalized varies in the loop).
        seadex_rg_normalized = normalize_rg(seadex_rg)

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
            episode_key = season_episode_key(seadex_ep_season, seadex_ep_episode)
            sonarr_ep = ctx.sonarr_by_key.get(episode_key)
            if sonarr_ep is None:
                continue

            # Get the matched Sonarr episode's file size
            sonarr_ep_size = sonarr_ep.episode_file.size if sonarr_ep.episode_file else None

            # Do the sizes match? A missing Sonarr file reports no
            # size, so guard against None == None reading as a match
            # when neither side actually has a size.
            size_match = sonarr_ep_size is not None and sonarr_ep_size == seadex_ep_size

            season_ep_str = f"S{seadex_ep_season:02d}E{seadex_ep_episode:02d}"

            # The file's effective identity, resolved once per entry: its own
            # tag, or - where a rename stripped it - the pick whose listed size
            # named it. Every url and sibling pick reads the same answer.
            identity = ctx.episode_identities.get(episode_key)
            sonarr_rg_normalized = identity.group if identity else None
            size_identified = identity is not None and identity.by_size

            # A mismatched group flags a download unless another recommended
            # release covers it. The coverage index is keyed by normalized
            # name, so compare the normalized name. The blanket test runs
            # first, unconditionally; the per-episode set only inside it.
            if sonarr_rg_normalized != seadex_rg_normalized and sonarr_rg_normalized not in ctx.coverage.blanket:
                # Avoid duplicating when another release already covers it
                covering = ctx.coverage.by_key.get(episode_key, frozenset())

                if sonarr_rg_normalized not in covering:
                    if ctx.debug_on:
                        # The raw on-disk tag, for log fidelity (debug only).
                        sonarr_rg = sonarr_ep.episode_file.release_group if sonarr_ep.episode_file else None
                        self.logger.debug(
                            f"SeaDex release group {seadex_rg} differs from "
                            f"{self.arr.capitalize()} release for "
                            f"{season_ep_str} ({sonarr_rg or 'no group'}) and no other "
                            f"recommended release covers it - will download {url}",
                        )

                    url_item.download = True

            else:
                if ctx.debug_on:
                    self.logger.debug(f"Found SeaDex match to {self.arr.capitalize()} for {season_ep_str}.")
                    if not size_match:
                        self.logger.debug(
                            f"Sizes are different: {sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
                        )
                    else:
                        self.logger.debug(f"Sizes match: {sonarr_ep_size}")

                # A size-identified episode stays OUT of the upgrade fold
                # below: its size matched by construction (that is what named
                # it), so its vote could only ever be "current" and would mask
                # a genuinely stale release around it.
                if not size_identified:
                    rg_matches[seadex_idx] = True

            # Now check against file size
            if size_match:
                size_matches[seadex_idx] = True

        # Of the episodes we matched by group: EVERY one at an unlisted size
        # means the whole release is a size upgrade to grab. (Partly stale
        # copies are the group verdicts' business - see `_group_verdicts`.)
        matched_sizes = list(compress(size_matches, rg_matches))
        if matched_sizes and not any(matched_sizes):
            self.logger.debug(f"File sizes all differ for release group {seadex_rg} - will download {url}")
            url_item.download = True
            url_item.upgrade = True

    def reduce_overlapping_downloads(
        self,
        seadex_dict: SeadexDict,
    ) -> PrivateOnlySkips:
        """Reduce overlapping flagged downloads down to a single release group.

        Where multiple preferred release groups cover the same files and the
        Arr doesn't already have any of them, we only want to grab one. We
        prefer a public release group and drop the private ones (private
        releases are never grabbed). If the only options are private, we
        record a warning SkipNotice and skip the title (without caching it as
        done) rather than grabbing a private release - unless an unflagged
        public group covering the same files rides along (a
        `private_releases: fallback` stand-in or a preferred public pick).
        Then the private groups are dropped with an INFO notice instead: a
        *preferred* public group is promoted (grabbed) when a private flag was
        a size-mismatch upgrade, and left alone when the Arr genuinely already
        owns its files. A fallback is never promoted over an owned copy of the
        preferred private release - those sets warn and hold (`stale_held`).

        After each set resolves, just-dropped public urls whose coverage no
        surviving url carries are re-flagged (group-atomic drops must not lose
        episodes). Finally, within each group, flagged urls carrying identical
        non-empty file-name sets (cross-seeded copies of one release) are
        deduped to the first.

        Mutates the download flags on seadex_dict in place and returns the
        private-only skip outcome (skipped flag, group names, notices to log).
        Skipped entirely in interactive mode, since hand-picking has already
        decided what to grab.
        """

        skips = PrivateOnlySkips()

        # In interactive mode, releases were explicitly hand-picked to
        # grab, so don't second-guess that choice by dropping any
        if self.interactive:
            return skips

        for same_files in get_same_files_groups(seadex_dict):
            # Only the release groups the Arr doesn't already have are flagged
            flagged = [rg for rg in same_files if _is_flagged(seadex_dict[rg])]
            if len(flagged) == 0:
                continue

            dropped = self._reduce_same_files_set(seadex_dict, same_files, flagged, skips)
            self._rescue_dropped_coverage(seadex_dict, same_files, dropped)

        # Within ONE group, flagged urls carrying the same files are the same
        # release cross-seeded (distinct infohashes = duplicate downloads, the
        # promotion branch above can flip several at once): keep the first,
        # unflag the rest. Identity compares file-size multisets - one listing
        # may fold files into a folder the other lists flat, and two per-cour
        # listings can even name their episodes identically, but the bytes
        # differ. An unsized listing can't prove identity, so it's never
        # deduped. Cross-group overlap is the same-files logic above.
        for rg_item in seadex_dict.values():
            seen: set[tuple[int, ...]] = set()
            for u in rg_item.urls.values():
                if not u.download or not u.size:
                    continue
                identity = tuple(sorted(u.size))
                if identity in seen:
                    # Upgrade clears with download, same as _unflag.
                    u.download = False
                    u.upgrade = False
                else:
                    seen.add(identity)

        return skips

    def _reduce_same_files_set(
        self,
        seadex_dict: SeadexDict,
        same_files: list[str],
        flagged: list[str],
        skips: PrivateOnlySkips,
    ) -> list[SeadexUrlItem]:
        """Resolve ONE same-files set down to a single keeper (or a skip).

        Appends any notices/skip state to `skips` and returns the urls this
        pass unflagged, for the coverage rescue.
        """

        dropped: list[SeadexUrlItem] = []

        public_flagged = [rg for rg in flagged if _is_public_group(seadex_dict[rg])]
        # The held-stale-not-owned marker: a size-mismatch flag means the Arr
        # holds the release at a STALE size (upgrade pending), so an unflagged
        # public group in this set is NOT owned and may be promoted in its
        # place. Without it, promotion would re-download owned content.
        upgrade_pending = any(u.upgrade for rg in flagged for u in seadex_dict[rg].urls.values())

        if len(public_flagged) == 0:
            # An unflagged public group in THIS same-files set can stand in
            # (fallback or preferred, a flagged one would have taken the keeper
            # branch below). A public group covering OTHER files doesn't excuse
            # dropping this set, so the gate is per-set, never per-entry.
            fallback_rides = any(_is_fallback_group(seadex_dict[rg]) for rg in same_files)
            if upgrade_pending:
                promoted = self._promote_public_alternative(seadex_dict, same_files)
                if promoted is not None:
                    self._drop_promoted_over(flagged, "private-only", promoted, skips)
                    for rg in flagged:
                        _unflag(seadex_dict[rg], dropped)
                    return dropped
            elif fallback_rides:
                # Unflagged with no size mismatch: the Arr genuinely already
                # owns the fallback's files.
                skips.fallback_covered = True
                skips.notices.append(
                    SkipNotice(
                        groups=list(flagged),
                        reason="private-only; a public fallback already covers these files",
                        severity=Severity.INFO,
                    ),
                )
                for rg in flagged:
                    _unflag(seadex_dict[rg], dropped)
                return dropped

            # The Arr has none of these release groups, private grabs are off,
            # and no promotable public url covers the same files. Don't grab a
            # private release, just record a skip notice and skip. Flag the skip
            # so the caller doesn't cache the title as done. A riding fallback
            # here means an owned-at-stale-size preferred pick it must not
            # replace (only reachable upgrade-pending, the soft-skip consumed
            # the other case): mark the stale hold for the summary row.
            # Invariant: a fallback substitute never replaces an owned copy of the
            # preferred private release - those sets hold and warn every run.
            if fallback_rides:
                reason = (
                    "private-only; your copy is outdated (its file size no longer matches the release) "
                    "and only a fallback covers it"
                )
                skips.stale_held = True
            else:
                reason = "private-only (private releases not supported)"
            skips.notices.append(
                SkipNotice(
                    groups=list(flagged),
                    reason=reason,
                    severity=Severity.WARNING,
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
                    self._drop_promoted_over(flagged, "remaining files private-only", promoted, skips)
                    for rg in flagged:
                        _unflag(seadex_dict[rg], dropped)
                    return dropped
                # Promotion refused with an unflagged fallback riding: an
                # owned-at-stale-size pick a fallback must not replace. Mark the
                # stale hold (no notice, the add-time private gate warns).
                if any(not _is_flagged(seadex_dict[rg]) and _is_fallback_group(seadex_dict[rg]) for rg in same_files):
                    skips.stale_held = True
            # Degrade to the first public group (the add-time gate then warns).
            keeper = public_flagged[0]

        # Note when the keeper is a non-preferred fallback standing in
        # for private preferred picks (a plain public-over-private keeper stays
        # a debug line, as ever).
        private_dropped = [rg for rg in flagged if not _is_public_group(seadex_dict[rg])]
        if private_dropped and _is_fallback_group(seadex_dict[keeper]):
            skips.notices.append(
                SkipNotice(
                    groups=private_dropped,
                    reason=f"private-only; falling back to {keeper}",
                    severity=Severity.INFO,
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
                f"Not downloading release group {rg}: release group {keeper} already covers the same files",
            )
            _unflag(seadex_dict[rg], dropped)

    @staticmethod
    def _promote_public_alternative(seadex_dict: SeadexDict, same_files: list[str]) -> str | None:
        """Flip on an unflagged public group covering this set. None if there is none.

        Never a fallback group - a substitute must not replace an owned stale
        copy of the preferred pick. The caller holds instead. Prefer a
        fully-public group - promotion only flips public urls, so a mixed
        group's private url would leave part of the set's coverage ungrabbed.
        """

        candidates = [
            rg
            for rg in same_files
            if not _is_flagged(seadex_dict[rg])
            and _is_public_group(seadex_dict[rg])
            and not _is_fallback_group(seadex_dict[rg])
        ]
        if not candidates:
            return None

        promoted = next((rg for rg in candidates if _all_public(seadex_dict[rg])), candidates[0])
        for u in seadex_dict[promoted].urls.values():
            if u.is_public:
                u.download = True
        return promoted

    @staticmethod
    def _drop_promoted_over(
        flagged: list[str],
        prefix: str,
        promoted: str,
        skips: PrivateOnlySkips,
    ) -> None:
        """The INFO notice for flagged groups a promoted group now stands in for.

        The caller unflags them. Promotion never picks a fallback group, so the
        verb is always "grabbing public alternative". The keeper flow owns
        "falling back to".
        """

        skips.notices.append(
            SkipNotice(
                groups=list(flagged),
                reason=f"{prefix}; grabbing public alternative {promoted}",
                severity=Severity.INFO,
            ),
        )

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
            self.logger.debug(f"Re-flagging {u.url}: no surviving release covers its episodes")
            u.download = True
            survivor_keys |= url_keys

    @staticmethod
    def get_any_to_download(seadex_dict: SeadexDict) -> bool:
        """Check if any torrents are marked as to download."""

        return any(url_item.download for rg_item in seadex_dict.values() for url_item in rg_item.urls.values())
