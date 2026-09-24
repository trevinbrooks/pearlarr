"""Pure episode-file statuses, the per-target snapshot an import checks, and the group trust its guard reads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import NamedTuple

from .manual_import import GuardFacts, OwnGroup, PendingImport, normalize_group, normalize_rg
from .placement_types import EpisodeIndex

type TrustPolicy = Mapping[str, frozenset[int] | None]
"""Normalized group -> the sizes that verify its files, or None to trust the group by name alone."""


class EpisodeFileStatus(Enum):
    """How an intended target episode's CURRENT Sonarr file relates to ours.

    One read of the episode list drives both invariants: never overwrite a
    recommended file, never skip an episode we intended to import.
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
        """True only when EVERY intended target already holds a recommended file.

        The "already imported / drop the record" signal. An UNKNOWN_GROUP or
        OTHER_GROUP file is NOT done (we still intend to import ours), so a
        present-but-unidentifiable file never makes us drop a record prematurely.
        """

        return bool(self.by_id) and all(s is EpisodeFileStatus.RECOMMENDED for s in self.by_id.values())

    def needing_import(self) -> set[int]:
        """The never-skip set: every intended id NOT already a recommended file.

        ABSENT / OTHER_GROUP / UNKNOWN_GROUP all need our import. Only
        RECOMMENDED is excluded (it is done and must not be overwritten).
        """

        return {ep_id for ep_id, status in self.by_id.items() if status is not EpisodeFileStatus.RECOMMENDED}


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus what counts as already ours.

    The episode index and the trust policy are gathered together, so consumers never mix state from two
    different polls.
    """

    episodes: EpisodeIndex
    """The fresh episode index."""

    trusted: TrustPolicy
    """The per-group trust policy (see `trusted_groups`). A group absent here is not recommended: its
    files are replaced."""

    owned_episode_sizes: Mapping[int, int] = MappingProxyType({})
    """Episode id -> the untagged file size the grab-time identification recorded. The claim is honored
    only while the file still sits at that size. Anything else untagged classifies as unidentifiable."""

    def statuses(self, target_ep_ids: list[int]) -> TargetStatuses:
        """Classify each intended target episode by its current on-disk file.

        Pure: reads only this snapshot's episode index and per-group trust
        policy (keyed via `normalize_group`). "Already imported" is decided
        HERE from the episode files, not from the queue, since Sonarr drops
        an imported item from its queue almost immediately.
        """

        statuses: dict[int, EpisodeFileStatus] = {}
        for ep_id in target_ep_ids:
            if ep_id in statuses:
                continue
            ep = self.episodes.by_id.get(ep_id)
            if ep is None or not ep.episode_file_id:
                statuses[ep_id] = EpisodeFileStatus.ABSENT
                continue
            group = ep.episode_file.release_group if ep.episode_file else None
            size = ep.episode_file.size if ep.episode_file else None
            if not group:
                # An untagged file still at the exact size the grab-time
                # identification recorded is a recommended copy. Anything else
                # untagged (a different file landed meanwhile, or no readable
                # file record at all) stays unidentifiable.
                statuses[ep_id] = (
                    EpisodeFileStatus.RECOMMENDED
                    if size is not None and size == self.owned_episode_sizes.get(ep_id)
                    else EpisodeFileStatus.UNKNOWN_GROUP
                )
                continue
            norm = normalize_group(group)
            if norm not in self.trusted:
                statuses[ep_id] = EpisodeFileStatus.OTHER_GROUP
                continue
            verify_sizes = self.trusted[norm]
            if verify_sizes is not None and size not in verify_sizes:
                # A trusted group at a size no current listing carries: the stale
                # copy this grab replaces, not our just-imported file.
                statuses[ep_id] = EpisodeFileStatus.OTHER_GROUP
            else:
                statuses[ep_id] = EpisodeFileStatus.RECOMMENDED
        return TargetStatuses(statuses)


@dataclass(frozen=True, slots=True)
class RecordSnapshot:
    """One poll's coherent view of every series a record spans, keyed by series id."""

    by_series: Mapping[int, EpisodeSnapshot]
    """Each series' same-poll snapshot."""

    indexes: Mapping[int, EpisodeIndex] = field(init=False)
    """Each series' fresh episode index, the placement windows' inputs (a view over `by_series`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_series", MappingProxyType(dict(self.by_series)))
        object.__setattr__(
            self,
            "indexes",
            MappingProxyType({series_id: snapshot.episodes for series_id, snapshot in self.by_series.items()}),
        )

    def series_of(self, ep_id: int) -> int | None:
        """The series whose index holds `ep_id` (Sonarr episode ids are global, so at most one does)."""

        return next((sid for sid, snapshot in self.by_series.items() if ep_id in snapshot.episodes.by_id), None)

    def statuses(self, target_ep_ids: Sequence[int]) -> TargetStatuses:
        """Classify each target under its series' snapshot; an id no index holds is ABSENT."""

        grouped: dict[int | None, list[int]] = {}
        for ep_id in dict.fromkeys(target_ep_ids):
            grouped.setdefault(self.series_of(ep_id), []).append(ep_id)
        by_id: dict[int, EpisodeFileStatus] = {}
        for series_id, ids in grouped.items():
            if series_id is None:
                by_id.update(dict.fromkeys(ids, EpisodeFileStatus.ABSENT))
            else:
                by_id.update(self.by_series[series_id].statuses(ids).by_id)
        return TargetStatuses({ep_id: by_id[ep_id] for ep_id in dict.fromkeys(target_ep_ids)})


def trusted_groups(
    guards: GuardFacts,
    own: OwnGroup,
    series_records: Sequence[PendingImport] = (),
) -> TrustPolicy:
    """One claim's per-group trust policy: group -> verifying sizes, or None for trust-by-name.

    The one home of the overwrite-guard composition, for grab time (no
    `series_records`) and import time (the series' pending records, which may
    include this record's own row, whose votes are no-ops) alike. The entry's
    verified-current pick groups and the series' other grabbed groups are
    trusted by name. A sibling's group is refused when THIS claim's plan
    judged it stale on disk (the copies being replaced must not ride back into
    protection on a sibling's vote). The record's OWN group joins last and
    unconditionally (it is the identity of the files being imported), but at
    the sizes its current listings carry (unioned across same-group records),
    so a stale same-group copy is told apart by size and replaced. No listed
    sizes means no size gate (the legacy trust-by-name behavior).
    """

    stale = {norm for g in guards.stale_groups if (norm := normalize_rg(g))}
    trusted: dict[str, frozenset[int] | None] = {norm: None for g in guards.entry_groups if (norm := normalize_rg(g))}
    own_norm = normalize_rg(own.release_group)
    own_sizes = set(own.sizes)
    for record in series_records:
        norm = normalize_rg(record.release_group)
        if norm is None:
            continue
        if norm == own_norm:
            own_sizes.update(record.release_sizes)
        if norm not in stale:
            trusted.setdefault(norm, None)
    if own_norm:
        trusted[own_norm] = frozenset(own_sizes) or None
    return trusted
