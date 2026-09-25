"""Pure episode-file statuses, the per-target snapshot an import checks, and the group trust its guard reads."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import NamedTuple, Self

from .manual_import import EntryClaim, GuardFacts, OwnGroup, PendingImport, normalize_group, normalize_rg
from .placement_types import EpisodeIndex

type TrustPolicy = Mapping[str, frozenset[int] | None]
"""Normalized group -> the sizes that verify its files, or None to trust the group by name alone."""


class EpisodeFileStatus(Enum):
    """How an intended target episode's CURRENT Sonarr file relates to ours.

    One read drives both invariants: never overwrite a recommended file, never skip an intended episode.
    """

    ABSENT = auto()
    """No file yet. Import ours."""

    RECOMMENDED = auto()
    """Already holds a file from a recommended group (ours, another torrent we grabbed for this series, or
    a group the entry's SeaDex picks carried at grab time), or an untagged file still at the exact size
    the grab-time identification recorded. It is done: do NOT overwrite it."""

    OTHER_GROUP = auto()
    """Holds a file from a non-recommended group, or our own group at a size no current listing carries
    (a stale copy this grab replaces). Import ours over it (the operator's intended replacement)."""

    UNKNOWN_GROUP = auto()
    """Holds a file with no parseable group that no recorded size identifies either. Import ours rather
    than trust an unidentifiable file as recommended."""


@dataclass(frozen=True, slots=True)
class TargetStatuses:
    """Each intended target episode's file status, with the two folds every consumer reads."""

    by_id: Mapping[int, EpisodeFileStatus]
    """One status per de-duplicated target id."""

    def all_done(self) -> bool:
        """True only when EVERY intended target already holds a recommended file (the drop-the-record signal).

        An UNKNOWN_GROUP or OTHER_GROUP file is NOT done: an unidentifiable file never drops a record early.
        """

        return bool(self.by_id) and all(s is EpisodeFileStatus.RECOMMENDED for s in self.by_id.values())

    def needing_import(self) -> set[int]:
        """The never-skip set: every intended id NOT already a recommended file (which must not be overwritten)."""

        return {ep_id for ep_id, status in self.by_id.items() if status is not EpisodeFileStatus.RECOMMENDED}


class GroupVotes(NamedTuple):
    """The groups a trust policy trusts beyond its guards: the torrent's own, then the series' other grabs."""

    own: OwnGroup
    siblings: Sequence[OwnGroup] = ()


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus what counts as already ours.

    The index and the trust policy are gathered together, so consumers never mix state from two polls.
    """

    episodes: EpisodeIndex
    """The fresh episode index."""

    trusted: TrustPolicy
    """The per-group trust policy (see `trusted_groups`). A group absent here is not recommended: its
    files are replaced."""

    owned_episode_sizes: Mapping[int, int] = MappingProxyType({})
    """Episode id -> the untagged file size the grab-time identification recorded. The claim is honored
    only while the file still sits at that size. Anything else untagged classifies as unidentifiable."""

    @classmethod
    def guarded(cls, episodes: EpisodeIndex, guards: GuardFacts, votes: GroupVotes) -> Self:
        """The snapshot under one claim's guard evidence: its trust policy and its recorded untagged sizes."""

        return cls(episodes, trusted_groups(guards, votes), guards.owned_sizes)

    def status_of(self, ep_id: int) -> EpisodeFileStatus:
        """Classify one intended target by its current on-disk file, decided HERE and not from the queue.

        Sonarr drops an imported item from its queue almost immediately, so only the episode files tell.
        """

        ep = self.episodes.by_id.get(ep_id)
        if ep is None or not ep.episode_file_id:
            return EpisodeFileStatus.ABSENT
        group = ep.episode_file.release_group if ep.episode_file else None
        size = ep.episode_file.size if ep.episode_file else None
        if not group:
            # An untagged file still at the size the grab-time identification recorded is a recommended copy.
            # Anything else untagged (a different file, or no readable file record) stays unidentifiable.
            if size is not None and size == self.owned_episode_sizes.get(ep_id):
                return EpisodeFileStatus.RECOMMENDED
            return EpisodeFileStatus.UNKNOWN_GROUP
        norm = normalize_group(group)
        if norm not in self.trusted:
            return EpisodeFileStatus.OTHER_GROUP
        verify_sizes = self.trusted[norm]
        if verify_sizes is not None and size not in verify_sizes:
            # A trusted group at a size no current listing carries: the stale copy this grab replaces.
            return EpisodeFileStatus.OTHER_GROUP
        return EpisodeFileStatus.RECOMMENDED

    def statuses(self, target_ep_ids: Sequence[int]) -> TargetStatuses:
        """Classify each de-duplicated intended target by its current on-disk file."""

        return TargetStatuses({ep_id: self.status_of(ep_id) for ep_id in dict.fromkeys(target_ep_ids)})


class Route(NamedTuple):
    """Where an episode id is judged: the claim whose guards apply, if one, and the series whose index holds it."""

    claim: EntryClaim | None
    series_id: int | None


def lone_unscoped_claims(claims: Sequence[EntryClaim]) -> dict[int, EntryClaim]:
    """The one unscoped claim of each series that has exactly one: it judges the series' ids outside every window."""

    unscoped_on = Counter(claim.series_id for claim in claims if not claim.ordered_episode_ids)
    return {
        claim.series_id: claim
        for claim in claims
        if not claim.ordered_episode_ids and unscoped_on[claim.series_id] == 1
    }


@dataclass(frozen=True, slots=True)
class RecordSnapshot:
    """One poll's coherent view of a record: each claimed series' snapshot, and each claim's own over it."""

    pending: PendingImport
    """The record the poll judges, whose claims route each target."""

    by_series: Mapping[int, EpisodeSnapshot]
    """Each claimed series' same-poll snapshot under the series' merged guard evidence (the routing fallback)."""

    by_claim: Mapping[int, EpisodeSnapshot]
    """Each claim's snapshot by AniList id: its series' same index under the claim's OWN guard evidence."""

    indexes: Mapping[int, EpisodeIndex] = field(init=False)
    """Each series' fresh episode index, the placement windows' inputs (a view over `by_series`)."""

    lone_unscoped: Mapping[int, EntryClaim] = field(init=False)
    """Each series' one unscoped claim where it has exactly one (see `lone_unscoped_claims`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_series", MappingProxyType(dict(self.by_series)))
        object.__setattr__(self, "by_claim", MappingProxyType(dict(self.by_claim)))
        object.__setattr__(self, "lone_unscoped", MappingProxyType(lone_unscoped_claims(self.pending.claims)))
        object.__setattr__(
            self,
            "indexes",
            MappingProxyType({series_id: snapshot.episodes for series_id, snapshot in self.by_series.items()}),
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
        return Route(self.lone_unscoped.get(series_id), series_id)

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

    def statuses(self, target_ep_ids: Sequence[int]) -> TargetStatuses:
        """Classify each target under the snapshot that judges it. An id no snapshot judges is ABSENT."""

        by_id: dict[int, EpisodeFileStatus] = {}
        for ep_id in dict.fromkeys(target_ep_ids):
            snapshot = self.snapshot_for(ep_id)
            by_id[ep_id] = EpisodeFileStatus.ABSENT if snapshot is None else snapshot.status_of(ep_id)
        return TargetStatuses(by_id)


def trusted_groups(guards: GuardFacts, votes: GroupVotes) -> TrustPolicy:
    """One claim's trust policy: entry picks and non-stale siblings by name, the own group last at its listed sizes.

    Same-group siblings union their sizes into the own group's (None = no size gate), so a stale copy is replaced.
    """

    stale = {norm for g in guards.stale_groups if (norm := normalize_rg(g))}
    trusted: dict[str, frozenset[int] | None] = {norm: None for g in guards.entry_groups if (norm := normalize_rg(g))}
    own_norm = normalize_rg(votes.own.release_group)
    own_sizes = set(votes.own.sizes)
    for sibling in votes.siblings:
        norm = normalize_rg(sibling.release_group)
        if norm is None:
            continue
        if norm == own_norm:
            own_sizes.update(sibling.sizes)
        if norm not in stale:
            trusted.setdefault(norm, None)
    if own_norm:
        trusted[own_norm] = frozenset(own_sizes) or None
    return trusted
