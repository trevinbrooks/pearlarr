"""A torrent's files placed under several windows. A window is the `TargetScope` of one entry the torrent is listed on.

Every window judges the WHOLE torrent, exactly as a record with one window does, so a numbered run keeps
its shape under every window (the remainder re-read as a fresh run could place onto a window the whole
run never fits). The verdicts merge in window order: the first window that places or refuses a file decides
it, so an earlier window's `HELD` or `MISNUMBERED` outranks a later window's exact placement (window order
beats pass order). Ids an earlier window placed count as used under each later window that admits them. A
file no window placed or refused takes the highest-ranked verdict the windows gave it (`_CLAIM_RANK`).
One rule beats window order: a file the special aliases placed or claimed under one window, and another window
placed on other ids without them, is `MISNUMBERED` (`_disputed`). The ids its deciding window placed it on still
count as used under the later windows, which can only leave another file unplaced, never misfile one.
"""

from collections.abc import Iterable, Mapping, Sequence
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


class WindowVerdict(NamedTuple):
    """One file's merged verdict and, when placed, the index into the windows of the one that placed it."""

    placement: Placement
    window_index: int | None


class WindowedAssignment(NamedTuple):
    """One merged verdict per distinct name in `PlacementBatch.to_place`, batch order."""

    verdicts: tuple[WindowVerdict, ...]

    @property
    def merged(self) -> EpisodeAssignment:
        """The single-window view: every verdict as `assign_episode_ids` returns it."""

        return EpisodeAssignment(tuple(v.placement for v in self.verdicts))

    def assigned_under(self, index: int) -> dict[str, list[int]]:
        """Name -> episode ids for the files the window at `index` placed."""

        return {v.placement.name: list(v.placement.ids) for v in self.verdicts if v.window_index == index}


# Highest wins. SKIPPED outranks FOREIGN and EXTRA, so a file skipped under any window is never excluded as another
# slice's or as an extra. ALIASED_ELSEWHERE sits between them: the aliases name the file as this series' special,
# so a window that reads it as foreign doesn't write it off. DUPLICATE outranks SKIPPED (a refused duplicate beats a
# file placed nowhere). EXTRA depends on each window's titles: lowest.
_CLAIM_RANK: Mapping[PlacementVerdict, int] = {
    PlacementVerdict.EXTRA: 0,
    PlacementVerdict.FOREIGN: 1,
    PlacementVerdict.ALIASED_ELSEWHERE: 2,
    PlacementVerdict.SKIPPED: 3,
    PlacementVerdict.DUPLICATE: 4,
}
_BY_ALIAS = frozenset({PlacementVerdict.ALTERNATE, PlacementVerdict.ALIASED_ELSEWHERE})
"""The verdicts the special aliases give: placed on the aliased special, or claimed for the window that holds it."""


def assign_across_windows(batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """Place a torrent's files under every window in order and merge the verdicts (see the module).

    One window is exactly one `assign_episode_ids` call over the same batch and scope. Under no
    windows at all every file is skipped.
    """

    names = batch.names
    decided: dict[str, WindowVerdict] = {}
    judged: dict[str, list[Placement]] = {name: [] for name in names}
    for index, window in enumerate(windows):
        # Ids earlier windows placed count as used here: the ones this window admits, all of them when unscoped.
        scope = window.using(i for v in decided.values() for i in v.placement.ids)
        for placement in assign_episode_ids(batch, scope).placements:
            judged[placement.name].append(placement)
            if placement.name in decided:
                continue
            if placement.verdict.placed:
                decided[placement.name] = WindowVerdict(placement, index)
            elif placement.verdict.refused:
                # A refusal decides a name as a placement does: no later window may place it by its numbers.
                decided[placement.name] = WindowVerdict(placement, None)
    return WindowedAssignment(tuple(_merged(name, judged[name], decided.get(name)) for name in names))


def _merged(name: str, judged: Sequence[Placement], decided: WindowVerdict | None) -> WindowVerdict:
    """One file's verdict over every window.

    `MISNUMBERED` when `_disputed`, else the deciding window's, else the top claim (`_CLAIM_RANK`), else `SKIPPED`.
    With no deciding window, every verdict the windows gave is a claim, so each one has a rank.
    """

    if _disputed(judged):
        return WindowVerdict(Placement(name, (), PlacementVerdict.MISNUMBERED), None)
    if decided is not None:
        return decided
    if not judged:
        return WindowVerdict(Placement(name, (), PlacementVerdict.SKIPPED), None)
    return WindowVerdict(max(judged, key=lambda claim: _CLAIM_RANK[claim.verdict]), None)


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
    """The names of `batch` the map does not cover, placed under `windows` with the mapped ids already used.

    Each window marks used only the mapped ids it admits, as with ids an earlier window placed, so one
    series' placements never turn off another series' count-based passes. Every window still gets the whole
    map as `TargetScope.seeded`.
    """

    leftover = [name for name in batch.to_place if name not in seeded]
    held = {name: tuple(ids) for name, ids in seeded.items()}
    mapped = [ep_id for ids in held.values() for ep_id in ids]
    narrowed = tuple(replace(window.using(mapped), seeded=held) for window in windows)
    return assign_across_windows(PlacementBatch(leftover, batch.parsed), narrowed)
