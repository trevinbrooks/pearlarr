"""Pure episode-file statuses, the per-target snapshot an import checks, and the group trust its guard reads."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import NamedTuple, Self

from .arr_models import SonarrEpisodeFile
from .pending_import import FileEpisodeMap, OwnedAtGrab, PendingImport, TitleClaim, TorrentGroup, sizes_by_episode
from .placement_types import EpisodeIndex
from .releases import normalize_group, normalize_rg

type TrustedGroups = Mapping[str, frozenset[int] | None]
"""Normalized group -> the sizes that verify its files, or None to trust the group by name alone."""


class EpisodeFileStatus(Enum):
    """How an intended target episode's CURRENT Sonarr file relates to ours.

    One read drives both invariants: never skip an intended episode, and never overwrite a recommended file that
    isn't a copy of our own.
    """

    ABSENT = auto()
    """No file yet. Import ours."""

    TRUSTED = auto()
    """Holds a file from a recommended group (ours, another torrent we grabbed for this series, or a group the entry's
    SeaDex picks carried at grab time), or an untagged file still at the size the grab recorded for it. It's done:
    never imported over, except a copy of our own beside an episode that needs the file (`import_ids`)."""

    OTHER_GROUP = auto()
    """Holds a file from a non-recommended group, or our own group at a size no current listing carries
    (a stale copy this grab replaces). Import ours over it (the operator's intended replacement)."""

    UNKNOWN_GROUP = auto()
    """Holds a file with no parseable group that no recorded size identifies either. Import ours rather
    than trust an unidentifiable file as recommended."""

    MISPLACED = auto()
    """Holds a trusted file at one of this torrent's listed sizes that is not the file intended on this episode
    (a batch imported under the wrong numbers). Import ours over it."""


class EpisodeFileFacts(NamedTuple):
    """A target's current file, read once: its status, and whether it's a copy of one of this torrent's files."""

    status: EpisodeFileStatus
    has_torrent_size: bool
    """The file is at one of this torrent's listed sizes. Size alone decides, not the group or the status."""
    has_torrent_size_no_group: bool = False
    """The file has no group and is a copy of the one intended here, or of any of this torrent's files when nothing
    is intended here: what a hand import of the torrent's own file leaves."""


# How a target with no file reads. `ImportSnapshot.status_of` also gives it for an id no snapshot covers.
_NO_FILE = EpisodeFileFacts(EpisodeFileStatus.ABSENT, has_torrent_size=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class StillNeeded:
    """The targets holding neither a recommended file nor an untagged copy of this torrent's, in target order."""

    missing: tuple[int, ...]
    """The ones holding no file (`EpisodeFileStatus.ABSENT`)."""

    untrusted: tuple[int, ...]
    """The ones holding a file."""


@dataclass(frozen=True, slots=True)
class EpisodeStatuses:
    """Each intended target's file status, plus the targets holding a copy of one of this torrent's files."""

    by_id: Mapping[int, EpisodeFileStatus]
    """One status per de-duplicated target id."""

    at_torrent_size: frozenset[int]
    """The targets whose current file is at one of this torrent's listed sizes (`TargetJudgment.own_copy`)."""

    at_torrent_size_no_group: frozenset[int] = frozenset()
    """The targets whose current file is an untagged copy of this torrent's (`TargetJudgment.untagged_copy`)."""

    @classmethod
    def for_episodes(
        cls,
        episode_ids: Iterable[int],
        intended_sizes: Mapping[int, int],
        facts_of: Callable[[int, int | None], EpisodeFileFacts],
    ) -> Self:
        """Judge each distinct target against the size `intended_sizes` puts on it."""

        by_episode = {ep_id: facts_of(ep_id, intended_sizes.get(ep_id)) for ep_id in dict.fromkeys(episode_ids)}
        return cls(
            {ep_id: facts.status for ep_id, facts in by_episode.items()},
            frozenset(ep_id for ep_id, facts in by_episode.items() if facts.has_torrent_size),
            frozenset(ep_id for ep_id, facts in by_episode.items() if facts.has_torrent_size_no_group),
        )

    def all_trusted(self) -> bool:
        """True only when every intended target holds a recommended file (the drop-the-record signal).

        An UNKNOWN_GROUP, OTHER_GROUP, or MISPLACED file is not done: only a recommended file drops a record.
        """

        return bool(self.by_id) and all(s is EpisodeFileStatus.TRUSTED for s in self.by_id.values())

    def still_needed(self) -> StillNeeded:
        """The targets a hand import has yet to fill, split by whether they hold a file."""

        still_needed = [
            (ep_id, status)
            for ep_id, status in self.by_id.items()
            if status is not EpisodeFileStatus.TRUSTED and ep_id not in self.at_torrent_size_no_group
        ]
        return StillNeeded(
            missing=tuple(ep_id for ep_id, status in still_needed if status is EpisodeFileStatus.ABSENT),
            untrusted=tuple(ep_id for ep_id, status in still_needed if status is not EpisodeFileStatus.ABSENT),
        )

    def all_done(self) -> bool:
        """Whether every target holds a recommended file or an untagged copy of this torrent's: a hand import done.

        Sonarr reads no group off a file name that carries none, so `all_done` alone never sees that hand import.
        """

        still_needed = self.still_needed()
        return bool(self.by_id) and not still_needed.missing and not still_needed.untrusted

    def needing_import(self) -> set[int]:
        """The never-skip set: every intended id NOT already holding a recommended file."""

        return {ep_id for ep_id, status in self.by_id.items() if status is not EpisodeFileStatus.TRUSTED}

    def import_ids(self, ep_ids: Sequence[int]) -> tuple[int, ...]:
        """Which of a file's episodes `ep_ids` to post it onto, in order, or none when no episode needs it.

        An episode holding a copy of our own goes along with a needing one, so Sonarr replaces the copy with the
        file instead of keeping both.
        """

        needing = self.needing_import()
        if not any(ep_id in needing for ep_id in ep_ids):
            return ()
        return tuple(ep_id for ep_id in ep_ids if ep_id in needing or ep_id in self.at_torrent_size)


class GrabbedGroups(NamedTuple):
    """The groups a trust policy trusts beyond its guards: the torrent's own, then the series' other grabs."""

    torrent_group: TorrentGroup
    siblings: Sequence[TorrentGroup] = ()


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus what counts as already ours.

    The index and the trust policy are gathered together, so consumers never mix state from two polls.
    """

    episodes: EpisodeIndex
    """The fresh episode index."""

    trusted_groups: TrustedGroups
    """The per-group trust policy (see `trusted_groups`). A group absent here is not recommended: its
    files are replaced."""

    torrent_group: TorrentGroup
    """The torrent's own release: a trusted file at one of its listed sizes is judged against the intended size."""

    owned_episode_sizes: Mapping[int, int] = MappingProxyType({})
    """Episode id -> the untagged file size the grab-time identification recorded. The claim is honored
    only while the file still sits at that size. Anything else untagged classifies as unidentifiable."""

    @classmethod
    def guarded(cls, episodes: EpisodeIndex, guards: OwnedAtGrab, grabbed_groups: GrabbedGroups) -> Self:
        """The snapshot under one claim's guard evidence: its trust policy and its recorded untagged sizes."""

        return cls(
            episodes=episodes,
            trusted_groups=trusted_groups(guards, grabbed_groups),
            torrent_group=grabbed_groups.torrent_group,
            owned_episode_sizes=guards.owned_sizes,
        )

    def _is_torrent_size(self, size: int | None) -> bool:
        """Whether a file at `size` is one of this torrent's files, going by size alone."""

        return size in self.torrent_group.sizes

    def _misplaced(self, size: int | None, intended_size: int | None) -> bool:
        """Whether a file at `size` is a file of this torrent's listed sizes that is not the one intended here."""

        return self._is_torrent_size(size) and intended_size is not None and size != intended_size

    def status_of(self, ep_id: int, intended_size: int | None) -> EpisodeFileFacts:
        """Judge one intended target by its current on-disk file: its status, and whether it's a copy of our own.

        Read from the episode files, never the queue: Sonarr drops an imported item from its queue almost at once.
        """

        ep = self.episodes.by_id.get(ep_id)
        if ep is None or not ep.episode_file_id:
            return _NO_FILE
        file = ep.episode_file
        if file is None:
            return EpisodeFileFacts(self._status(ep_id, file, intended_size), has_torrent_size=False)
        has_torrent_size = self._is_torrent_size(file.size)
        has_torrent_size_no_group = (
            has_torrent_size and not file.release_group and not self._misplaced(file.size, intended_size)
        )
        return EpisodeFileFacts(self._status(ep_id, file, intended_size), has_torrent_size, has_torrent_size_no_group)

    def _status(self, ep_id: int, file: SonarrEpisodeFile | None, intended_size: int | None) -> EpisodeFileStatus:
        """Classify a target's current file. `file` is None when Sonarr gives the episode a file id but no record."""

        group = file.release_group if file else None
        size = file.size if file else None
        if not group:
            # An untagged file still at the size the grab-time identification recorded is ours, recommended unless
            # it is another of our files. Anything else untagged (a different file, no file record) is unidentifiable.
            if size is not None and size == self.owned_episode_sizes.get(ep_id):
                misplaced = self._misplaced(size, intended_size)
                return EpisodeFileStatus.MISPLACED if misplaced else EpisodeFileStatus.TRUSTED
            return EpisodeFileStatus.UNKNOWN_GROUP
        norm = normalize_group(group)
        if norm not in self.trusted_groups:
            return EpisodeFileStatus.OTHER_GROUP
        verify_sizes = self.trusted_groups[norm]
        if verify_sizes is not None and size not in verify_sizes:
            # A trusted group at a size no current listing carries: the stale copy this grab replaces.
            return EpisodeFileStatus.OTHER_GROUP
        if self._misplaced(size, intended_size):
            # Byte-identical to another of our files (a same-files sibling's too): the episode still lacks its own.
            return EpisodeFileStatus.MISPLACED
        return EpisodeFileStatus.TRUSTED

    def statuses(self, file_map_episode_ids: Sequence[int], intended_sizes: Mapping[int, int]) -> EpisodeStatuses:
        """Classify each de-duplicated target by its current file, judged against its intended size."""

        return EpisodeStatuses.for_episodes(file_map_episode_ids, intended_sizes, self.status_of)


class Route(NamedTuple):
    """Where an episode id is judged: the claim whose guards apply, if one, and the series whose index holds it."""

    claim: TitleClaim | None
    series_id: int | None


def sole_claims_without_window(claims: Sequence[TitleClaim]) -> dict[int, TitleClaim]:
    """The one unscoped claim of each series that has exactly one: it judges the series' ids outside every window."""

    without_window_on = Counter(claim.series_id for claim in claims if not claim.ordered_episode_ids)
    return {
        claim.series_id: claim
        for claim in claims
        if not claim.ordered_episode_ids and without_window_on[claim.series_id] == 1
    }


@dataclass(frozen=True, slots=True)
class ImportSnapshot:
    """One poll's coherent view of a record: each claimed series' snapshot, and each claim's own over it."""

    pending: PendingImport
    """The record the poll judges, whose claims route each target."""

    by_series: Mapping[int, EpisodeSnapshot]
    """Each claimed series' same-poll snapshot under the series' merged guard evidence (the routing fallback)."""

    by_claim: Mapping[int, EpisodeSnapshot]
    """Each claim's snapshot by AniList id: its series' same index under the claim's OWN guard evidence."""

    sole_claim_without_window: Mapping[int, TitleClaim] = field(init=False)
    """Each series' one unscoped claim where it has exactly one (see `lone_unscoped_claims`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_series", MappingProxyType(dict(self.by_series)))
        object.__setattr__(self, "by_claim", MappingProxyType(dict(self.by_claim)))
        object.__setattr__(
            self, "sole_claim_without_window", MappingProxyType(sole_claims_without_window(self.pending.claims))
        )

    def series_of(self, ep_id: int) -> int | None:
        """The series whose index holds `ep_id` (Sonarr episode ids are global, so at most one does)."""

        return next((sid for sid, snapshot in self.by_series.items() if ep_id in snapshot.episodes.by_id), None)

    def route(self, ep_id: int) -> Route:
        """The claim and series that judge `ep_id`.

        The first scoped claim naming it decides. Otherwise the series whose index holds it does, under its
        unscoped claim when it has exactly one (two unscoped claims on a series merge), else under its merged evidence.
        """

        for claim in self.pending.claims:
            if ep_id in claim.ordered_episode_ids:
                return Route(claim, claim.series_id)
        series_id = self.series_of(ep_id)
        if series_id is None:
            return Route(None, None)
        return Route(self.sole_claim_without_window.get(series_id), series_id)

    def snapshot_for(self, ep_id: int) -> EpisodeSnapshot | None:
        """The snapshot that judges `ep_id`: its route's claim's own, else its route's series'.

        None when no claim and no index holds it.
        """

        route = self.route(ep_id)
        if route.claim is not None and (own := self.by_claim.get(route.claim.al_id)) is not None:
            return own
        return None if route.series_id is None else self.by_series.get(route.series_id)

    def preowned(self, ep_id: int) -> bool:
        """Whether the grab judging `ep_id` found it owned: its route's claim says, else any claim (merged fallback)."""

        route = self.route(ep_id)
        claims = (route.claim,) if route.claim is not None else self.pending.claims
        return any(ep_id in claim.preowned_episode_ids for claim in claims)

    def status_of(self, ep_id: int, intended_size: int | None) -> EpisodeFileFacts:
        """Judge `ep_id` under the snapshot its route picks, or as ABSENT when no snapshot holds it."""

        snapshot = self.snapshot_for(ep_id)
        return _NO_FILE if snapshot is None else snapshot.status_of(ep_id, intended_size)

    def file_map_statuses(self, file_map_episode_ids: Sequence[int], file_map: FileEpisodeMap) -> EpisodeStatuses:
        """Judge each de-duplicated target as `judge` does, against the listed size of the file `file_map` puts on it."""

        return EpisodeStatuses.for_episodes(
            file_map_episode_ids, sizes_by_episode(file_map, self.pending.sizes_by_name), self.status_of
        )


def trusted_groups(guards: OwnedAtGrab, grabbed_groups: GrabbedGroups) -> TrustedGroups:
    """One claim's trust policy: entry picks and non-stale siblings by name, the own group last at its listed sizes.

    Same-group siblings union their sizes into the own group's (None = no size gate), so a stale copy is replaced.
    """

    outdated = {norm for g in guards.stale_groups if (norm := normalize_rg(g))}
    trusted: dict[str, frozenset[int] | None] = {norm: None for g in guards.entry_groups if (norm := normalize_rg(g))}
    own_norm = normalize_rg(grabbed_groups.torrent_group.release_group)
    own_sizes = set(grabbed_groups.torrent_group.sizes)
    for sibling in grabbed_groups.siblings:
        norm = normalize_rg(sibling.release_group)
        if norm is None:
            continue
        if norm == own_norm:
            own_sizes.update(sibling.sizes)
        if norm not in outdated:
            trusted.setdefault(norm, None)
    if own_norm:
        trusted[own_norm] = frozenset(own_sizes) or None
    return trusted
