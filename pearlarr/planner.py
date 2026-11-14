"""The grab planner: which SeaDex releases to grab.

`GrabPlanner` is near-pure: it consumes the shaped `seadex_dict`, the
Arr's files and an optional episode list, and returns a
`GrabPlan`. It flips each release's `to_grab` in place and reports the
private-only skip outcome (`GrabPlan.skips`: *what to log* and *what was
skipped*) as data, rather than reaching into the orchestrator's run state or
its log formatter.
"""

import logging
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import compress
from typing import NamedTuple

from .arr_models import ArrFiles, SonarrEpisode, index_episodes_by_key
from .config import Arr
from .output import Severity
from .pairs import unambiguous
from .pending_import import OwnedAtGrab, SizeMatchedEpisode
from .releases import (
    EpisodeKey,
    GroupReleases,
    Release,
    ReleaseEpisode,
    SeadexDict,
    normalize_group,
    season_episode_key,
)


@dataclass
class PrivateOnlyNotice:
    """A release dropped for being private, for the caller to log.

    Rendered by the orchestrator as `"<groups> <reason>"` on a `skipped`
    detail line. A private-only skip logs at WARNING. A drop with a public
    group grabbed in its place logs at INFO.
    """

    groups: list[str]
    reason: str
    severity: Severity = Severity.WARNING


@dataclass
class PrivateOnlyNotices:
    """The private-only skip outcome of `reduce_overlapping_grabs`."""

    warned: bool = False
    """True when at least one same-files set was dropped with a warning: none of its groups to grab was
    public, and no public group was promoted in their place."""
    groups: list[str] = field(default_factory=list[str])
    """The dropped groups' names, for the run summary."""
    notices: list[PrivateOnlyNotice] = field(default_factory=list[PrivateOnlyNotice])
    """What to log for the drop."""
    outdated: bool = False
    """A private-only skip where the Arr's copy of the preferred private release is outdated and only a
    fallback could stand in. A fallback never replaces a copy of the preferred release."""


@dataclass
class GrabPlan:
    """The grab planner's output.

    Carries the annotated dict and the private-only skip outcome (`skips`) so
    the orchestrator can log it, name it in the summary, and decide whether to
    cache the title.
    """

    seadex_dict: SeadexDict
    """The same dict passed in, each release's `to_grab` set in place."""
    skips: PrivateOnlyNotices
    """The private-only skip outcome from `reduce_overlapping_grabs`, merged into run state by the caller."""
    guards: OwnedAtGrab
    """The overwrite-guard evidence the import inherits whole (see `OwnedAtGrab`):
    the size-verified owned groups, the positively outdated ones, and the owned
    `(episode id, size)` claims of files with no group. Only Sonarr's import reads it."""


def _no_group_counter(untagged_sizes: Iterable[int]) -> Counter[int] | None:
    """The size multiset of the arr's files with no group, or None when it can never claim ownership.

    Built once per entry (the ownership checks run per release). None when empty,
    or when any size is non-positive - a zero-byte failed copy proves nothing
    and stays grabbable, so the whole multiset is disqualified with it.
    """

    counter = Counter(untagged_sizes)
    if not counter or any(size <= 0 for size in counter):
        return None
    return counter


def _owns_no_group_copy(counter: Counter[int] | None, listed_sizes: Iterable[int]) -> bool:
    """Whether every file the arr has with no group belongs to this release, by exact size.

    Identifies an owned copy whose group tag a rename stripped. Containment runs
    arr -> release, never the reverse: a release also carries files the arr never
    has (subtitles, fonts, creditless extras), so requiring every LISTED size
    in the arr would only ever match a single-file release. Multisets, so two
    files with no group at one size need two listed twins. `counter` is the
    `_no_group_counter` multiset (None short-circuits: nothing ownable in the arr).
    """

    return counter is not None and counter <= Counter(listed_sizes)


def _doubled_sizes(
    listed: Mapping[EpisodeKey, int], arr_sizes: Mapping[EpisodeKey, int], sizes: Collection[int]
) -> list[int]:
    """The release's `sizes` the arr has on more of the `listed` keys than the release places them on (possibly none).

    An `arr_sizes` key absent from `listed` (an episode two listed files size differently) counts on neither side.
    """

    placed = Counter(listed.values())
    arr_counts = Counter(size for key, size in arr_sizes.items() if key in listed)
    return sorted(size for size, count in arr_counts.items() if size in sizes and placed[size] < count)


def get_episode_keys(
    all_episodes: Iterable[ReleaseEpisode],
) -> set[tuple[int | None, int | None]]:
    """Build the set of (season, episode) keys an episode list covers.

    Reduces a release's parsed episode list to the set of (season, episode)
    pairs it contains, so different SeaDex release groups can be compared by
    what files they cover.
    """

    return {(ep.season, ep.episode) for ep in all_episodes}


def get_same_files_groups(seadex_dict: SeadexDict) -> list[list[str]]:
    """Group SeaDex release groups that cover exactly the same files.

    Groups are keyed by their placed episode coverage and collapse only when the sets are identical: a
    season batch and one cour overlap yet cover different files. Groups with no parsing at all (Radarr
    movies) share one key. A group none of whose files placed on the title is kept on its own, since its
    coverage is unproven and a duplicate grab beats silently dropped content. Two groups a name number pass
    placed onto the same window reduce to one, so a wrong placement drops the real release, the same
    hazard the import carries.
    """

    # The grouping key is one of three shapes: a shared "all cover one movie"
    # sentinel str, a per-group "couldn't parse" sentinel tuple, or the parsed
    # episode-coverage frozenset that equates groups covering identical files.
    grouped: dict[
        str | tuple[str, str] | frozenset[tuple[int | None, int | None]],
        list[str],
    ] = {}
    for rg, group_releases in seadex_dict.items():
        all_episodes = group_releases.all_episodes

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


def _group_to_grab(group_releases: GroupReleases) -> bool:
    return any(u.to_grab for u in group_releases.releases.values())


def _is_public_group(group_releases: GroupReleases) -> bool:
    return any(u.is_public for u in group_releases.releases.values())


def _is_fallback_group(group_releases: GroupReleases) -> bool:
    return any(u.is_fallback for u in group_releases.releases.values())


def _all_public(group_releases: GroupReleases) -> bool:
    return all(u.is_public for u in group_releases.releases.values())


def _grabs_all_public(group_releases: GroupReleases) -> bool:
    # No private release to grab (the add-time gate skips those).
    return all(u.is_public for u in group_releases.releases.values() if u.to_grab)


def _drop_group_grabs(group_releases: GroupReleases, dropped: list[Release]) -> None:
    """Drop a whole group's grabs, recording the releases this pass actually dropped."""

    for u in group_releases.releases.values():
        if u.to_grab:
            dropped.append(u)
        u.drop_grab()


@dataclass(slots=True)
class _SameFilesSet:
    """One same-files coverage set mid-reduction: the groups, the subset to grab, and this pass's drops."""

    names: list[str]
    """Every group in the set, in first-seen order (`get_same_files_groups`)."""

    to_grab: list[str]
    """The subset still marked to grab when the reduction began."""

    dropped: list[Release] = field(default_factory=list[Release])
    """The releases whose grabs the reduction dropped, limited to THIS set - the coverage rescue's input."""


def _promotion_notice(groups: list[str], prefix: str, promoted: str) -> PrivateOnlyNotice:
    """The INFO notice for groups to grab that a promoted group now stands in for.

    The caller drops their grabs. Promotion never picks a fallback group, so the
    verb is always "grabbing public alternative". The keeper flow owns
    "falling back to".
    """

    return PrivateOnlyNotice(
        groups=list(groups),
        reason=f"{prefix}; grabbing public alternative {promoted}",
        severity=Severity.INFO,
    )


class EpisodeCoverage(NamedTuple):
    """Which sibling groups cover each episode - the matcher's duplicate-avoidance index.

    Consulted when the group of a file the arr has mismatches the release being decided:
    a covering group means another preferred release already accounts for
    the episode, so no duplicate grab. Membership is by normalized name; a
    group whose name normalizes to None is indexed NOWHERE - an unidentifiable
    SeaDex group must never read as covering a file with no group.
    """

    blanket: frozenset[str]
    """Groups with a release whose parses key into nothing here: coverage unprovable, so they cover every
    episode unconditionally."""

    by_key: Mapping[EpisodeKey, frozenset[str]]
    """The groups whose parsed releases cover each (season, episode) Sonarr actually has."""


def episode_coverage(
    seadex_dict: SeadexDict,
    sonarr_by_key: Mapping[EpisodeKey, SonarrEpisode],
) -> EpisodeCoverage:
    """Index which SeaDex groups cover each episode, by normalized name.

    The release episodes are the grab-time placement's, so every key is one of the
    title's own episodes. `sonarr_by_key` is that index, built once by the
    caller and shared with the per-episode match loop in
    `_plan_by_episodes`. Built for a single group too: a torrent left for a hand import
    reads its own group's placed episodes here.
    """

    blanket: set[str] = set()
    by_key: dict[EpisodeKey, set[str]] = {}
    for seadex_rg, group_releases in seadex_dict.items():
        # Index by the normalized name so the membership checks in
        # `plan` are case- and dash-insensitive.
        seadex_rg_normalized = normalize_group(seadex_rg)
        if seadex_rg_normalized is None:
            continue

        for release in group_releases.releases.values():
            seadex_episodes = release.episodes

            # A release none of whose files placed on this title proves nothing
            # about coverage, so its group covers everything rather than nothing.
            if not seadex_episodes:
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


class _FileReleaseGroup(NamedTuple):
    """The effective release identity of one episode file the arr has."""

    group: str | None
    """The normalized group: the file's own tag, or the group a listed size named."""
    by_size: bool
    """Identified by exact listed size, not a tag (see `_file_release_groups`)."""
    size: int | None
    """The file's size (always the listed size for a `by_size` identity)."""


class _FileReleaseGroups(NamedTuple):
    """`_file_release_groups`' two views of the arr's files - deliberately OVERLAPPING.

    `groups` holds every identified file (by group tag or by size);
    `no_group_sizes` holds every file with no group, the size-identified ones
    included, because the unparsed-release matcher compares that multiset
    whole. The overlap is the size-identified files - do not join the views.
    """

    groups: dict[EpisodeKey, _FileReleaseGroup]
    """Each identified file's effective release identity."""

    no_group_sizes: dict[EpisodeKey, int]
    """The size of every file the arr has with no group (zero/unknown read as 0)."""


def _file_release_groups(
    seadex_dict: SeadexDict,
    sonarr_by_key: Mapping[EpisodeKey, SonarrEpisode],
) -> _FileReleaseGroups:
    """Resolve the effective identity of each file the arr has: its group tag, or the group its size names.

    A library rename scheme can strip the release-group tag off a file the arr
    already has, leaving it unidentifiable by name. A POSITIVE exact size match
    against a release's listed episode still says whose copy it is (a zero or
    unknown size proves nothing, so a failed copy stays grabbable).

    Entry-wide on purpose: ownership is a property of the FILE, so every release of
    the entry must read it the same way. Deciding it per release instead would let
    the release that matched skip its grab while a sibling release still grabbed
    the same episodes. The first release to claim an episode wins - a tie means two
    releases list it at the same size, where either answer reads as owned.
    """

    identities: dict[EpisodeKey, _FileReleaseGroup] = {}
    # Every file the arr has with no group. A zero/unknown size reads as 0: it can never
    # size-match an identity below, and in the whole-copy multiset it rules out
    # ownership (a failed copy proves nothing and stays grabbable). Almost
    # always empty (a library whose files have groups), which skips the loop over the releases.
    untagged: dict[EpisodeKey, int] = {}
    for key, sonarr_ep in sonarr_by_key.items():
        arr_file = sonarr_ep.episode_file
        if arr_file is None:
            continue
        group = normalize_group(arr_file.release_group)
        if group is not None:
            identities[key] = _FileReleaseGroup(group, by_size=False, size=arr_file.size)
        else:
            untagged[key] = arr_file.size or 0
    if not untagged:
        return _FileReleaseGroups(identities, untagged)

    for seadex_rg, group_releases in seadex_dict.items():
        normalized = normalize_group(seadex_rg)
        if normalized is None:
            continue
        for release in group_releases.releases.values():
            # A torrent left for a hand import placed nothing: a file with no group at one of its listed sizes
            # is its, wherever it sits.
            if release.left_for_hand_import:
                for key, size in untagged.items():
                    if key not in identities and size > 0 and size in release.file_sizes:
                        identities[key] = _FileReleaseGroup(normalized, by_size=True, size=size)
                continue
            for seadex_ep in release.episodes:
                if seadex_ep.season is None or seadex_ep.episode is None:
                    continue
                key = season_episode_key(seadex_ep.season, seadex_ep.episode)
                if key in identities or seadex_ep.size <= 0 or untagged.get(key) != seadex_ep.size:
                    continue
                identities[key] = _FileReleaseGroup(normalized, by_size=True, size=seadex_ep.size)
    return _FileReleaseGroups(identities, untagged)


@dataclass(frozen=True, slots=True)
class _PlanContext:
    """Per-entry invariants for the release match loop.

    Computed once per entry in `plan` and shared by both
    per-release matchers. Nothing here changes inside the loop (only each
    release's `to_grab`/`is_upgrade` fields flip).
    """

    arr_groups_label: str
    """The Arr's release-group names pre-rendered for the no-episodes matcher's
    debug lines (insertion order preserved; empty on a normal INFO run)."""
    arr_sizes_by_norm: Mapping[str, list[int]]
    """The sizes of the Arr's files with a group, merged under normalized group names. Built once per entry rather
    than per release."""
    any_release_owned: bool
    """The Arr already owns some release: by group tag, or as the copy with no group
    a release's listed sizes identified whole (so a sibling release is never
    grabbed over a copy only a rename left unrecognizable)."""
    sonarr_by_key: dict[EpisodeKey, SonarrEpisode]
    file_release_groups: Mapping[EpisodeKey, _FileReleaseGroup]
    """The effective release identity of each of the arr's files (see `_file_release_groups`)."""
    untagged_counter: Counter[int] | None
    """The size multiset of the arr's files with no group (Radarr's `None` key, or Sonarr's
    episode files with no group), built by `_no_group_counter` - the unparsed-release
    ownership input. None when no file without a group can claim ownership."""
    coverage: EpisodeCoverage
    """Which sibling groups cover each episode (see `episode_coverage`)."""
    has_ep_list: bool
    debug_on: bool


def _window_owned(ctx: _PlanContext, seadex_norm: str | None, listed_sizes: Collection[int]) -> bool:
    """Whether every episode of the window has the group's file at a listed size, or a covering group's file."""

    for key in ctx.sonarr_by_key:
        identity = ctx.file_release_groups.get(key)
        if identity is None or identity.group is None:
            return False
        covering = ctx.coverage.by_key.get(key, ())
        if identity.group == seadex_norm:
            # The group's own file at a listed size, or on an episode another release of the group placed.
            if identity.size not in listed_sizes and seadex_norm not in covering:
                return False
        # A sibling that placed nothing here covers every episode, as the ordinary match reads it.
        elif identity.group not in ctx.coverage.blanket and identity.group not in covering:
            return False
    return True


class _GroupOwnership(NamedTuple):
    """The plan's size-evidence ownership per group, feeding `OwnedAtGrab` by name."""

    owned: tuple[str, ...]
    """Groups whose every attributed file in the arr sits at a listed size (vacuously, when the arr has none)."""

    outdated: tuple[str, ...]
    """Groups with an attributed file at a size no sized release carries."""


def _group_ownership(seadex_dict: SeadexDict, ctx: _PlanContext) -> _GroupOwnership:
    """Sort each group's copy in the arr by size evidence: `(owned, outdated)` names.

    File-centric: a group is OWNED only when every file the arr has attributed to
    it (by tag, or by the size identification) sits at a size some release of the
    group lists - vacuously when the arr has none of it. Any attributed file at
    a size no sized release carries makes it OUTDATED. A release with no sizes is
    no evidence either way, so an all-blind group with files in the arr lands in
    NEITHER list: unprotected at import, but never voted an upgrade. Whole-group
    ownership is conservative in the overwrite direction - one outdated file exits
    the group from the never-overwrite set entirely.
    """

    cover: dict[str | None, set[int]] = {}
    for rg, item in seadex_dict.items():
        listed = cover.setdefault(normalize_group(rg), set())
        for release in item.releases.values():
            # Release-level and per-episode sizes both count (an episode parse can
            # carry sizes the release's file list doesn't); zero proves nothing.
            listed.update(s for s in release.file_sizes if s > 0)
            listed.update(ep.size for ep in release.episodes if ep.size > 0)

    # The arr's files each group must answer for. With an episode list the
    # per-episode identities attribute them (size-identified files included);
    # without one, the Arr's tag-keyed size index is the whole vocabulary.
    attributed: dict[str | None, list[int | None]] = {}
    if ctx.has_ep_list:
        for identity in ctx.file_release_groups.values():
            attributed.setdefault(identity.group, []).append(identity.size)
    else:
        attributed = {norm: list(sizes) for norm, sizes in ctx.arr_sizes_by_norm.items()}

    owned: list[str] = []
    outdated: list[str] = []
    for rg in seadex_dict:
        norm = normalize_group(rg)
        files = attributed.get(norm, [])
        listed = cover.get(norm, set())
        if not files:
            owned.append(rg)
        elif not listed:
            continue
        elif all(size is not None and size in listed for size in files):
            owned.append(rg)
        elif any(size is not None and size not in listed for size in files):
            outdated.append(rg)
        # A group whose only disagreements are unknown sizes stays unverifiable.
    return _GroupOwnership(tuple(owned), tuple(outdated))


class GrabPlanner:
    """Decides which SeaDex releases to grab for one title.

    Constructed once per Arr run with the arr it plans for and whether the run
    is interactive. `plan` takes the already-shaped
    `seadex_dict` plus the Arr's files and returns a `GrabPlan`. The
    planner keeps a logger only for the per-release debug lines. The
    operator-facing private-only skip is returned as a `PrivateOnlyNotice`, never
    logged here.
    """

    def __init__(
        self,
        *,
        arr: Arr,
        interactive: bool,
        logger: logging.Logger,
    ) -> None:
        self.arr = arr
        self.interactive = interactive
        self.logger = logger

    def plan(
        self,
        seadex_dict: SeadexDict,
        arr_files: ArrFiles,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> GrabPlan:
        """Filter torrents by release group.

        This is either an episode-by-episode for the Sonarr
        case where we can parse episodes, or a more blunt
        hammer just checking against anything for Radarr
        and weirdly named TV
        """

        # The sizes of the Arr's files with a group, merged under normalized group names, built
        # once per entry and shared by the name-overlap check here and the
        # per-release matchers. A group key is truthy, so the None guard only
        # narrows the type (`normalize_group` returns a str for truthy input).
        arr_sizes_by_norm: dict[str, list[int]] = {}
        for arr_rg, sizes in arr_files.by_group.items():
            if (norm := normalize_group(arr_rg)) is not None:
                arr_sizes_by_norm.setdefault(norm, []).extend(sizes)

        # Whether any SeaDex release group matches any Arr release tag - by
        # normalized name, matching the per-episode path's comparison.
        seadex_keys = {normalize_group(rg) for rg in seadex_dict}
        any_release_owned = not seadex_keys.isdisjoint(arr_sizes_by_norm)

        # Index the Sonarr episodes by (season, episode) once, shared by both
        # the coverage index below and the per-episode match loop: looking up a
        # parsed SeaDex (season, episode) is then an O(1) dict op rather than a
        # fresh search of the whole list.
        sonarr_by_key = index_episodes_by_key(ep_list or [])

        # Which sibling groups cover each episode, reusing the index above
        coverage = episode_coverage(seadex_dict, sonarr_by_key)

        # The effective identity of each of the arr's files - its tag, or the group whose
        # listed size names a file with no group - decided once for the whole entry
        # so every release reads the same ownership.
        identities, untagged_by_key = _file_release_groups(seadex_dict, sonarr_by_key)

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

        # Every file the arr has with no group, reduced to its ownership multiset once
        # per entry. Exactly one source per arr - Radarr's no-group sizes, or the
        # Sonarr identity pass (`sonarr_files` keeps `no_group` empty,
        # pinned) - a file fed through both would count twice and could then
        # never be owned. A release whose listed sizes contain the whole
        # multiset owns the copy, which counts as an overlap with the arr's files - a
        # sibling release must stay ungrabbed exactly as it would had the copy kept its tag.
        # None (the common library whose files have groups) skips the per-release compares entirely.
        untagged_counter = _no_group_counter((*arr_files.no_group, *untagged_by_key.values()))
        any_release_owned = any_release_owned or (
            untagged_counter is not None
            and any(
                _owns_no_group_copy(untagged_counter, release.file_sizes)
                for group_releases in seadex_dict.values()
                for release in group_releases.releases.values()
            )
        )

        ctx = _PlanContext(
            # Pre-rendered so the un-normalized names stay out of the
            # matchers' decisions (they feed only debug lines).
            arr_groups_label=arr_files.groups_label() if debug_on else "",
            arr_sizes_by_norm=arr_sizes_by_norm,
            any_release_owned=any_release_owned,
            sonarr_by_key=sonarr_by_key,
            file_release_groups=identities,
            untagged_counter=untagged_counter,
            coverage=coverage,
            has_ep_list=ep_list is not None,
            debug_on=debug_on,
        )

        for seadex_rg, group_releases in seadex_dict.items():
            self.logger.debug(f"Filtering for release group {seadex_rg}")

            for release in group_releases.releases.values():
                # A torrent left for a hand import is decided by what its window has, never by its numbers.
                if release.left_for_hand_import:
                    self._plan_left_for_hand_import(ctx, seadex_rg, release)
                    continue

                # A release none of whose files placed on this title (names with no numbers, a
                # sequence waiting on a parse, another season's torrent) is decided by release group and size.
                if not release.episodes:
                    self._plan_by_group_and_size(ctx, seadex_rg, release)
                    continue

                self._plan_by_episodes(ctx, seadex_rg, release)

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring a public group)
        skips = self.reduce_overlapping_grabs(seadex_dict=seadex_dict)

        ownership = _group_ownership(seadex_dict, ctx)
        return GrabPlan(
            seadex_dict=seadex_dict,
            skips=skips,
            guards=OwnedAtGrab(
                entry_groups=ownership.owned,
                stale_groups=ownership.outdated,
                # Resolved to (episode id, size) pairs here so the pending import updates
                # inherit the exact files the grab decision just called owned -
                # and can re-verify them by size before honoring the claim.
                owned_episodes=tuple(
                    SizeMatchedEpisode(eid, identity.size)
                    for key, identity in identities.items()
                    if identity.by_size and identity.size and (eid := sonarr_by_key[key].id)
                ),
            ),
        )

    def _plan_by_group_and_size(
        self,
        ctx: _PlanContext,
        seadex_rg: str,
        release: Release,
    ) -> None:
        """Decide a single release with no parsed episodes, by release group + size.

        Flips `release.to_grab` in place. The blunt path used for
        Radarr and weirdly named TV: if the group isn't in the Arr's files
        (and nothing overlaps) grab it. If it is, grab it only when the file
        sizes are disjoint. A copy a rename left with no group is identified by size
        instead (`_owns_no_group_copy`).
        """

        url = release.url

        # Match by the normalized name (like the per-episode path). The merged
        # size index was built once per entry (see _PlanContext). A blank group
        # name is no name - it can only match by owning the copy with no group below.
        seadex_norm = normalize_group(seadex_rg)
        if seadex_norm is not None and seadex_norm in ctx.arr_sizes_by_norm:
            # The group matches: fall through to a size comparison. A release
            # with no sizes carries no evidence either way, so the arr's copy
            # stands (a vacuous disjoint must not read as an upgrade).
            seadex_file_sizes = release.file_sizes
            arr_file_sizes = ctx.arr_sizes_by_norm[seadex_norm]

            if seadex_file_sizes and set(seadex_file_sizes).isdisjoint(arr_file_sizes):
                if ctx.debug_on:
                    self.logger.debug(
                        f"SeaDex release group {seadex_rg} in {self.arr.capitalize()} releases: "
                        f"{ctx.arr_groups_label}, but file sizes do not match - will download {url}",
                    )

                release.plan_grab(upgrade=True)

            elif ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} in {self.arr.capitalize()} releases: "
                    f"{ctx.arr_groups_label}, and no listed size contradicts the on-disk copy",
                )
        elif _owns_no_group_copy(ctx.untagged_counter, release.file_sizes):
            # A rename stripped the group tag: the arr's files with no group
            # (Radarr's None key, or Sonarr's episode files with no group) all
            # belong to this release by size, so the arr owns it.
            if ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases: "
                    f"{ctx.arr_groups_label}, but the untagged files on disk are all "
                    f"this release's - not flagging {url}",
                )
        elif not ctx.any_release_owned:
            if ctx.debug_on:
                self.logger.debug(
                    f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases: "
                    f"{ctx.arr_groups_label} - will download {url}",
                )

            release.plan_grab()
        elif ctx.debug_on:
            # Group absent, but the Arr already owns another SeaDex-preferred
            # group's release covering these files - nothing to grab.
            self.logger.debug(
                f"SeaDex release group {seadex_rg} not in {self.arr.capitalize()} releases, but another "
                f"SeaDex group already overlaps them - not flagging {url}",
            )

    def _plan_by_episodes(
        self,
        ctx: _PlanContext,
        seadex_rg: str,
        release: Release,
    ) -> None:
        """Decide a single release against its placed episodes, per episode.

        Flips `release.to_grab` in place. For each episode the placement put
        a SeaDex file on we check whether the release group matches and whether
        the file sizes match. A release-group mismatch with no covering
        alternative, or an all-sizes mismatch among the rg-matched episodes,
        marks it to grab. A file the arr has with no group that the entry's sizes
        identified (`_file_release_groups`) matches by that group instead of by
        its missing tag.

        A listed size the arr has from this group on more episodes than the release places
        it on marks the release to grab as an upgrade (the arr has it, not as listed), so a file
        imported under the wrong numbers gets its missing files grabbed. Files
        that only swapped episodes are not marked by this check: a renumbering
        looks the same.
        """

        # At this point, we need an episode list from Sonarr. Only an absent list
        # skips: an empty one is never passed in production (the strategy returns first).
        if not ctx.has_ep_list:
            self.logger.debug(
                "Skipping per-episode check: no Sonarr episode list available",
            )
            return

        url = release.url
        seadex_episodes = release.episodes

        # For each episode the placement put this release on, check whether the release group matches and
        # whether the file sizes match. A release-group mismatch with no alternative marks it to grab,
        # and so does every size mismatching.

        rg_matches = [False] * len(seadex_episodes)
        size_matches = [False] * len(seadex_episodes)
        # Fixed per call - normalize once, not once per parsed episode (only
        # sonarr_rg_normalized varies in the loop).
        seadex_rg_normalized = normalize_group(seadex_rg)
        # Per matched episode: each size the release places on it, and the size the arr has from this group there.
        listed_pairs: list[tuple[EpisodeKey, int]] = []
        arr_size_by_key: dict[EpisodeKey, int] = {}

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
            # re-searching the whole list for every parsed episode
            episode_key = season_episode_key(seadex_ep_season, seadex_ep_episode)
            sonarr_ep = ctx.sonarr_by_key.get(episode_key)
            if sonarr_ep is None:
                continue
            listed_pairs.append((episode_key, seadex_ep_size))

            # Get the matched Sonarr episode's file size
            sonarr_ep_size = sonarr_ep.episode_file.size if sonarr_ep.episode_file else None

            # Do the sizes match? A missing Sonarr file reports no
            # size, so guard against None == None reading as a match
            # when neither side actually has a size.
            size_match = sonarr_ep_size is not None and sonarr_ep_size == seadex_ep_size

            season_ep_str = f"S{seadex_ep_season:02d}E{seadex_ep_episode:02d}"

            # The file's effective identity, resolved once per entry: its own
            # tag, or - where a rename stripped it - the group whose listed size
            # named it. Every release and its siblings read the same answer.
            identity = ctx.file_release_groups.get(episode_key)
            sonarr_rg_normalized = identity.group if identity else None
            by_size = identity is not None and identity.by_size

            # A mismatched group marks the release to grab unless another preferred
            # release covers it. The coverage index is keyed by normalized
            # name, so compare the normalized name. The blanket test runs
            # first, unconditionally; the per-episode set only inside it.
            if sonarr_rg_normalized != seadex_rg_normalized and sonarr_rg_normalized not in ctx.coverage.blanket:
                # Avoid duplicating when another release already covers it
                covering = ctx.coverage.by_key.get(episode_key, frozenset())

                if sonarr_rg_normalized not in covering:
                    if ctx.debug_on:
                        # The arr's raw tag, for log fidelity (debug only).
                        sonarr_rg = sonarr_ep.episode_file.release_group if sonarr_ep.episode_file else None
                        self.logger.debug(
                            f"SeaDex release group {seadex_rg} differs from "
                            f"{self.arr.capitalize()} release for "
                            f"{season_ep_str} ({sonarr_rg or 'no group'}) and no other "
                            f"recommended release covers it - will download {url}",
                        )

                    release.plan_grab()

            else:
                if ctx.debug_on:
                    self.logger.debug(f"Found SeaDex match to {self.arr.capitalize()} for {season_ep_str}.")
                    if not size_match:
                        self.logger.debug(
                            f"Sizes are different: {sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
                        )
                    else:
                        self.logger.debug(f"Sizes match: {sonarr_ep_size}")

                # A size-identified episode stays OUT of the upgrade check
                # below: its size matched by construction (that is what named
                # it), so its vote could only ever be "owned" and would mask
                # a genuinely outdated release around it.
                if not by_size:
                    rg_matches[seadex_idx] = True

                if (
                    seadex_rg_normalized is not None
                    and sonarr_rg_normalized == seadex_rg_normalized
                    and sonarr_ep_size is not None
                ):
                    arr_size_by_key[episode_key] = sonarr_ep_size

            # Now check against file size
            if size_match:
                size_matches[seadex_idx] = True

        # Of the episodes we matched by group: EVERY one at an unlisted size
        # means the whole release is a size upgrade to grab. (Partly outdated
        # copies are the group ownership's business - see `_group_ownership`.)
        matched_sizes = list(compress(size_matches, rg_matches))
        if matched_sizes and not any(matched_sizes):
            self.logger.debug(f"File sizes all differ for release group {seadex_rg} - will download {url}")
            release.plan_grab(upgrade=True)

        # An episode two listed files claim at different sizes is left out: the import cannot pick between them.
        doubled = _doubled_sizes(unambiguous(listed_pairs), arr_size_by_key, release.file_sizes)
        if doubled:
            self.logger.debug(
                f"{url} lists size(s) {doubled} on fewer episodes than {seadex_rg} holds them, will download"
            )
            release.plan_grab(upgrade=True)

    def _plan_left_for_hand_import(
        self,
        ctx: _PlanContext,
        seadex_rg: str,
        release: Release,
    ) -> None:
        """Decide a torrent the placement left for a hand import (`SkipReason.left_for_hand_import`), by what its window has.

        Flips `release.to_grab` in place. Nothing is grabbed once every window episode has its file at a listed
        size or a covering release's (`_window_owned`), the shape a hand import leaves. Otherwise it is marked
        to grab (an upgrade where the arr has the group's files), and the grab skips it and lists it for a hand
        import.
        """

        if not ctx.has_ep_list:
            self.logger.debug("Skipping the refused-pack check: no Sonarr episode list available")
            return

        url = release.url
        seadex_norm = normalize_group(seadex_rg)
        if _window_owned(ctx, seadex_norm, release.file_sizes):
            if ctx.debug_on:
                self.logger.debug(
                    f"Every episode of the window holds {seadex_rg}'s file or a covering release's: not flagging {url}"
                )
            return
        self.logger.debug(f"{seadex_rg} was refused by the placement and is not in place: will download {url}")
        release.plan_grab(upgrade=seadex_norm in ctx.arr_sizes_by_norm)

    def reduce_overlapping_grabs(
        self,
        seadex_dict: SeadexDict,
    ) -> PrivateOnlyNotices:
        """Reduce overlapping grabs down to a single release group.

        Where multiple preferred release groups cover the same files and the
        Arr doesn't already have any of them, we only want to grab one. We
        prefer a public release group and drop the private ones (private
        releases are never grabbed). If the only options are private, we
        record a warning `PrivateOnlyNotice` and skip the private releases (leaving
        the title uncached) rather than grabbing a private release. The one way out: when
        a private grab is an upgrade over the arr's files (an outdated size, or a file
        on too many episodes) and a *preferred* public group not marked to grab covers
        the same files, that group is promoted (grabbed) with an INFO notice.
        A fallback is never promoted over the arr's copy of the preferred
        private release - those sets warn and skip (`outdated`).

        After each set resolves, just-dropped public releases whose coverage no
        surviving release carries are marked to grab again (group-atomic drops must not lose
        episodes). Finally, within each group, releases to grab carrying identical
        non-empty file-name sets (cross-seeded copies of one release) are
        deduped to the first.

        Mutates the `to_grab` marks on seadex_dict in place and returns the
        private-only skip outcome (skipped flag, group names, notices to log).
        Skipped entirely in interactive mode, since hand-picking has already
        decided what to grab.
        """

        skips = PrivateOnlyNotices()

        # In interactive mode, releases were explicitly hand-picked to
        # grab, so don't second-guess that choice by dropping any
        if self.interactive:
            return skips

        for names in get_same_files_groups(seadex_dict):
            # Only the release groups the Arr doesn't already have are marked to grab
            to_grab = [rg for rg in names if _group_to_grab(seadex_dict[rg])]
            if len(to_grab) == 0:
                continue

            same_files = _SameFilesSet(names=names, to_grab=to_grab)
            self._reduce_same_files_set(seadex_dict, same_files, skips)
            self._rescue_dropped_coverage(seadex_dict, same_files)

        # Within ONE group, releases to grab carrying the same files are the same
        # release cross-seeded (distinct infohashes = duplicate grabs, the
        # promotion branch above can flip several at once): keep the first,
        # drop the rest's grabs. Identity compares file-size multisets - one release
        # may put files into a folder the other lists flat, and two per-cour
        # releases can even name their episodes identically, but the bytes
        # differ. An unsized release can't prove identity, so it's never
        # deduped. Cross-group overlap is the same-files logic above.
        for group_releases in seadex_dict.values():
            seen: set[tuple[int, ...]] = set()
            for u in group_releases.releases.values():
                if not u.to_grab or not u.file_sizes:
                    continue
                identity = tuple(sorted(u.file_sizes))
                if identity in seen:
                    u.drop_grab()
                else:
                    seen.add(identity)

        return skips

    def _reduce_same_files_set(
        self,
        seadex_dict: SeadexDict,
        same_files: _SameFilesSet,
        skips: PrivateOnlyNotices,
    ) -> None:
        """Resolve ONE same-files set down to a single keeper (or a skip).

        Appends any notices/skip state to `skips` and records the releases whose grabs this
        pass dropped on `same_files.dropped`, for the coverage rescue.
        """

        to_grab = same_files.to_grab
        public_to_grab = [rg for rg in to_grab if _is_public_group(seadex_dict[rg])]
        # The has-but-not-owned marker: an upgrade means the Arr has the
        # release wrong (an outdated size, or a file on too many episodes), so a
        # public group in this set not marked to grab is NOT owned and may be promoted in
        # its place. Without it, promotion would grab owned content again.
        any_upgrade = any(u.is_upgrade for rg in to_grab for u in seadex_dict[rg].releases.values())

        if len(public_to_grab) == 0:
            # Only a preferred public group not marked to grab in THIS same-files set
            # can be promoted in its place (one marked to grab would have taken the
            # keeper branch below). A public group covering OTHER files never
            # stands in, so the search is per-set, never per-entry.
            if any_upgrade:
                promoted = self._promote_public_alternative(seadex_dict, same_files)
                if promoted is not None:
                    skips.notices.append(_promotion_notice(to_grab, "private-only", promoted))
                    self._drop_all_grabs(seadex_dict, same_files)
                    return

            # Private grabs are off and no promotable public release covers the
            # same files. Don't grab a private release, just record a skip
            # notice and skip. Flag the skip so the caller doesn't cache the
            # title. An upgrade with a fallback in the set means the arr's copy
            # of a preferred release (an outdated size, or misplaced files) the
            # fallback must not replace: mark the outdated skip for the summary row.
            # Invariant: a fallback never replaces an owned copy of the preferred private release. Those same-files
            # groups grab nothing and warn every run.
            if any_upgrade and any(_is_fallback_group(seadex_dict[rg]) for rg in same_files.names):
                reason = (
                    "private-only; your copy is outdated (its file size no longer matches the release) "
                    "and only a fallback covers it"
                )
                skips.outdated = True
            else:
                reason = "private-only (private releases not supported)"
            skips.notices.append(
                PrivateOnlyNotice(
                    groups=list(to_grab),
                    reason=reason,
                    severity=Severity.WARNING,
                ),
            )
            skips.warned = True
            skips.groups.extend(to_grab)
            self._drop_all_grabs(seadex_dict, same_files)
            return

        # Keep the first public release group whose releases to grab are all
        # addable: a mixed group's private release to grab is skipped at add time,
        # losing the coverage only it carries, so a fully-addable group wins
        # over it.
        keeper = next((rg for rg in public_to_grab if _grabs_all_public(seadex_dict[rg])), None)
        if keeper is None:
            # No fully-addable group: when the private grabs are upgrades over
            # the arr's files, a public group covering this set that isn't marked to grab still grabs
            # cleanly, so promote it rather than degrading to an add-time skip.
            if any_upgrade:
                promoted = self._promote_public_alternative(seadex_dict, same_files)
                if promoted is not None:
                    skips.notices.append(_promotion_notice(to_grab, "remaining files private-only", promoted))
                    self._drop_all_grabs(seadex_dict, same_files)
                    return
                # Promotion found no group while a fallback not marked to grab is in the set: the
                # arr's outdated copy of a preferred release, which a fallback must not replace. Mark the
                # outdated skip (no notice, the add-time private gate warns).
                if any(
                    not _group_to_grab(seadex_dict[rg]) and _is_fallback_group(seadex_dict[rg])
                    for rg in same_files.names
                ):
                    skips.outdated = True
            # Degrade to the first public group (the add-time gate then warns).
            keeper = public_to_grab[0]

        # Note when the keeper is a non-preferred fallback standing in
        # for private preferred releases (a plain public-over-private keeper stays
        # a debug line, as ever).
        private_dropped = [rg for rg in to_grab if not _is_public_group(seadex_dict[rg])]
        if private_dropped and _is_fallback_group(seadex_dict[keeper]):
            skips.notices.append(
                PrivateOnlyNotice(
                    groups=private_dropped,
                    reason=f"private-only; falling back to {keeper}",
                    severity=Severity.INFO,
                ),
            )

        self._drop_losers(seadex_dict, same_files, keeper)

    @staticmethod
    def _drop_all_grabs(seadex_dict: SeadexDict, same_files: _SameFilesSet) -> None:
        """Drop the grabs of every group of the set, recording the drops."""

        for rg in same_files.to_grab:
            _drop_group_grabs(seadex_dict[rg], same_files.dropped)

    def _drop_losers(
        self,
        seadex_dict: SeadexDict,
        same_files: _SameFilesSet,
        keeper: str,
    ) -> None:
        """Drop the grabs of every group but the keeper, recording the drops."""

        for rg in same_files.to_grab:
            if rg == keeper:
                continue

            self.logger.debug(
                f"Not downloading release group {rg}: release group {keeper} already covers the same files",
            )
            _drop_group_grabs(seadex_dict[rg], same_files.dropped)

    @staticmethod
    def _promote_public_alternative(seadex_dict: SeadexDict, same_files: _SameFilesSet) -> str | None:
        """Mark to grab a public group covering this set that isn't marked yet. None if there is none.

        Never a fallback group - a fallback must not replace the arr's outdated
        copy of the preferred release. The caller skips instead. Prefer a
        fully-public group - promotion only marks public releases, so a mixed
        group's private release would leave part of the set's coverage ungrabbed.
        """

        candidates = [
            rg
            for rg in same_files.names
            if not _group_to_grab(seadex_dict[rg])
            and _is_public_group(seadex_dict[rg])
            and not _is_fallback_group(seadex_dict[rg])
        ]
        if not candidates:
            return None

        promoted = next((rg for rg in candidates if _all_public(seadex_dict[rg])), candidates[0])
        for u in seadex_dict[promoted].releases.values():
            if u.is_public:
                u.plan_grab()
        return promoted

    def _rescue_dropped_coverage(
        self,
        seadex_dict: SeadexDict,
        same_files: _SameFilesSet,
    ) -> None:
        """Mark to grab again the just-dropped public releases whose coverage no survivor carries.

        A group-atomic drop can lose episodes: two mixed groups with equal
        coverage unions land in one set, and dropping the loser's grabs wholesale
        drops its public release's unique coverage. Those releases were marked to grab by the
        matcher (the Arr provably lacks their files) before this pass dropped
        them, so marking them again can't grab owned content. Limited to THIS
        set's just-dropped releases only - never ones the matcher left unmarked - and a release
        with no placed episodes (movies, unplaced groups) has no coverage
        vocabulary, so it is never rescued.
        """

        rescuable = [u for u in same_files.dropped if u.is_public and u.episodes]
        if not rescuable:
            return

        # What the set will actually obtain: surviving releases to grab the add-time
        # gate accepts (private releases are skipped there, so they never count).
        survivor_keys = {
            key
            for rg in same_files.names
            for u in seadex_dict[rg].releases.values()
            if u.to_grab and u.is_public
            for key in get_episode_keys(u.episodes)
        }
        for u in rescuable:
            release_keys = get_episode_keys(u.episodes)
            if release_keys <= survivor_keys:
                continue
            self.logger.debug(f"Re-flagging {u.url}: no surviving release covers its episodes")
            u.plan_grab()
            survivor_keys |= release_keys

    @staticmethod
    def any_to_grab(seadex_dict: SeadexDict) -> bool:
        """Check if any releases are marked to grab."""

        return any(
            release.to_grab for group_releases in seadex_dict.values() for release in group_releases.releases.values()
        )
