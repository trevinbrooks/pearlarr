"""A torrent's files placed under each entry it's listed on, one window per entry.

Every window places every file, not just the ones earlier windows left, because a numbered run cut short would read
as a new run and could land wrong. Then the windows merge in order:

- The first window to place or refuse a file decides it, and its ids are used under the later windows.
- A file no window decides takes its strongest claim (`_CLAIM_RANK`).
- A file the special aliases claim under one window but its numbers place elsewhere is always `MISNUMBERED`.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple

from .manual_import import EntryClaim, FileEpisodeMap
from .placement_types import (
    EpisodeAssignment,
    EpisodeIndex,
    ListingEvidence,
    Placement,
    PlacementBatch,
    PlacementVerdict,
    TargetScope,
)
from .placer import assign_episode_ids


class WindowedAssignment(NamedTuple):
    """A torrent placed under its windows: the merged verdicts, and each window's own."""

    merged: EpisodeAssignment
    """One verdict per file the stored map doesn't cover, in batch order."""
    windows: tuple[EpisodeAssignment, ...]
    """What the placer returned under each window, in window order."""

    def refuted_under(self, index: int) -> bool:
        """Whether a file this window refuted by its title is still `TITLE_REFUTED` after the merge."""

        refuted = PlacementVerdict.TITLE_REFUTED
        return self._left_as(refuted, {p.name for p in self.windows[index].placements if p.verdict is refuted})

    def misnumbered_under(self, index: int) -> bool:
        """Whether a file this window placed, refused, or claimed by alias ended up `MISNUMBERED`.

        Files the window wrote off (another entry's, an extra, a skip) don't count.
        """

        own = {
            p.name
            for p in self.windows[index].placements
            if p.verdict.placed or p.verdict.refused or p.verdict in _BY_ALIAS
        }
        return self._left_as(PlacementVerdict.MISNUMBERED, own)

    def _left_as(self, verdict: PlacementVerdict, names: Collection[str]) -> bool:
        """Whether the merge left any of `names` with `verdict`."""

        return any(p.verdict is verdict and p.name in names for p in self.merged.placements)


# A file no window decided ends with its highest-ranked claim. EXTRA is lowest because it rests on one window's
# titles. SKIPPED beats FOREIGN and EXTRA, so a file some window couldn't place is never written off, and
# ALIASED_ELSEWHERE beats FOREIGN because the aliases name the file as this series' special. TITLE_REFUTED is
# highest, so a file whose title contradicts its numbers stays a hand import.
_CLAIM_RANK: Mapping[PlacementVerdict, int] = {
    PlacementVerdict.EXTRA: 0,
    PlacementVerdict.FOREIGN: 1,
    PlacementVerdict.ALIASED_ELSEWHERE: 2,
    PlacementVerdict.SKIPPED: 3,
    PlacementVerdict.DUPLICATE: 4,
    PlacementVerdict.TITLE_REFUTED: 5,
}
_BY_ALIAS = frozenset({PlacementVerdict.ALTERNATE, PlacementVerdict.ALIASED_ELSEWHERE})
"""The verdicts the special aliases give: placed on the aliased special, or claimed for the window that holds it."""


def _merged(name: str, judged: Sequence[Placement], decided: Placement | None) -> Placement:
    """One file's verdict over every window.

    With no deciding window, every verdict the windows gave is a claim, so `_CLAIM_RANK` ranks each one.
    """

    if _disputed(judged):
        return Placement(name, (), PlacementVerdict.MISNUMBERED)
    if decided is not None:
        return decided
    if not judged:
        return Placement(name, (), PlacementVerdict.SKIPPED)
    return max(judged, key=lambda claim: _CLAIM_RANK[claim.verdict])


def _disputed(judged: Iterable[Placement]) -> bool:
    """Whether the aliases claimed the file under one window while another placed it on other ids by its numbers.

    A placement by numbers agrees only with an `ALTERNATE` on the same ids, since an `ALIASED_ELSEWHERE` has none.
    """

    by_alias = {placement.ids for placement in judged if placement.verdict in _BY_ALIAS}
    if not by_alias:
        return False
    unaliased = {
        placement.ids for placement in judged if placement.verdict.placed and placement.verdict not in _BY_ALIAS
    }
    return bool(unaliased - by_alias)


def windows_of(
    claims: Iterable[EntryClaim], indexes: Mapping[int, EpisodeIndex], listing: ListingEvidence
) -> tuple[TargetScope, ...]:
    """One window per claim, in claim order. `indexes` must hold every claim's series.

    All windows share `listing`, but a size match only counts under a window on that episode's series.
    """

    return tuple(
        TargetScope(list(claim.ordered_episode_ids), indexes[claim.series_id], names=claim.names, listing=listing)
        for claim in claims
    )


def place_leftover(seeded: FileEpisodeMap, batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """The files the stored map doesn't cover, placed under each window in turn, verdicts merged.

    Under each window, the mapped ids and the ids of files earlier windows decided are used, where the window admits
    them. Mapped files aren't placed again, but the map and their parses still count as evidence. With no windows,
    every file is skipped.
    """

    held = {name: tuple(ids) for name, ids in seeded.items()}
    leftover = PlacementBatch([name for name in batch.to_place if name not in held], batch.parsed)
    names = leftover.names
    used = [ep_id for ids in held.values() for ep_id in ids]
    decided: dict[str, Placement] = {}
    judged: dict[str, list[Placement]] = {name: [] for name in names}
    assignments: list[EpisodeAssignment] = []
    for window in windows:
        assignments.append(assignment := assign_episode_ids(leftover, replace(window.using(used), seeded=held)))
        for placement in assignment.placements:
            judged[placement.name].append(placement)
            # A refusal decides the file too, so no later window can place it by its numbers.
            if placement.name not in decided and (placement.verdict.placed or placement.verdict.refused):
                decided[placement.name] = placement
                used.extend(placement.ids)
    merged = EpisodeAssignment(tuple(_merged(name, judged[name], decided.get(name)) for name in names))
    return WindowedAssignment(merged, tuple(assignments))
