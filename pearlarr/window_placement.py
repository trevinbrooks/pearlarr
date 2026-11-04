"""A torrent's files placed under several windows. A window is the `TargetScope` of one entry the torrent is listed on.

Every window judges the WHOLE torrent, exactly as a record with one window does, so a numbered run keeps
its shape under every window (the remainder re-read as a fresh run could place onto a window the whole
run never fits). The verdicts merge in window order: the first window that places or refuses a file decides
it, so an earlier window's `HELD` or `MISNUMBERED` outranks a later window's exact placement (window order
beats pass order). The ids a deciding window placed a file on count as used under each later window that admits
them. A file no window placed or refused takes the highest-ranked verdict the windows gave it (`_CLAIM_RANK`).
One rule beats window order: a file the special aliases placed or claimed under one window, and another window
placed on other ids without them, is `MISNUMBERED` (`_disputed`). The ids its deciding window placed it on still
count as used under the later windows, which can only leave another file unplaced, never misfile one.
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
    """A torrent's files placed under several windows: the merged verdicts, and each window's own."""

    merged: EpisodeAssignment
    """One verdict per distinct name the stored map doesn't cover, batch order, merged over the windows."""
    windows: tuple[EpisodeAssignment, ...]
    """Each window's `assign_episode_ids` result, in window order."""

    def refuted_under(self, index: int) -> bool:
        """Whether the window at `index` gave `TITLE_REFUTED` to a file the merge left `TITLE_REFUTED`."""

        refuted = PlacementVerdict.TITLE_REFUTED
        return self._left_as(refuted, {p.name for p in self.windows[index].placements if p.verdict is refuted})

    def misnumbered_under(self, index: int) -> bool:
        """Whether the merge left `MISNUMBERED` a file the window at `index` placed, refused, or claimed by alias.

        A file that window wrote off (another slice's, an extra, a skip) is no misnumbered file of its own.
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


# Highest wins. SKIPPED outranks FOREIGN and EXTRA, so a file skipped under any window is never excluded as another
# slice's or as an extra. ALIASED_ELSEWHERE sits between them: the aliases name the file as this series' special,
# so a window that reads it as foreign doesn't write it off. DUPLICATE outranks SKIPPED (a refused duplicate beats a
# file placed nowhere). EXTRA depends on each window's titles: lowest. TITLE_REFUTED rejects only its own window's
# reading: the top claim, so a file no window places stays a hand import.
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

    `MISNUMBERED` when `_disputed`, else the deciding window's, else the top claim (`_CLAIM_RANK`), else `SKIPPED`.
    With no deciding window, every verdict the windows gave is a claim, so each one has a rank.
    """

    if _disputed(judged):
        return Placement(name, (), PlacementVerdict.MISNUMBERED)
    if decided is not None:
        return decided
    if not judged:
        return Placement(name, (), PlacementVerdict.SKIPPED)
    return max(judged, key=lambda claim: _CLAIM_RANK[claim.verdict])


def _disputed(judged: Iterable[Placement]) -> bool:
    """Whether the aliases spoke for the file under one window and another window placed it elsewhere without them.

    An `ALIASED_ELSEWHERE` claim carries no ids, so a placement without the aliases stands only where an
    `ALTERNATE` put the file too.
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
    """One window per claim, in claim order, over its series' index (`indexes` holds every claim's series).

    Every window gets the same `listing`. A file's size match only counts in windows of that episode's series.
    """

    return tuple(
        TargetScope(list(claim.ordered_episode_ids), indexes[claim.series_id], names=claim.names, listing=listing)
        for claim in claims
    )


def place_leftover(seeded: FileEpisodeMap, batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """The names of `batch` the stored map doesn't cover, placed under every window in order, verdicts merged.

    Each window is one `assign_episode_ids` call over those names. The mapped ids are used under it, and so are the
    ids of each file an earlier window decided, where the window admits them (every id, when unscoped). Every window
    still gets the whole map as `TargetScope.seeded`, and the parses of the whole batch. Under no windows at all every
    file is skipped. The module docstring says how the verdicts merge.
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
            # A refusal decides a name as a placement does: no later window may place it by its numbers.
            if placement.name not in decided and (placement.verdict.placed or placement.verdict.refused):
                decided[placement.name] = placement
                used.extend(placement.ids)
    merged = EpisodeAssignment(tuple(_merged(name, judged[name], decided.get(name)) for name in names))
    return WindowedAssignment(merged, tuple(assignments))
