"""A torrent's files placed under several windows: one `TargetScope` per entry the torrent is listed on.

Every window judges the WHOLE torrent, exactly as its own record does under one window, so a run
keeps its shape under every window (a remainder re-read as a fresh run could index a window the whole
run never could). The verdicts merge by window order: the first window that places or holds a name
decides it, so an earlier window's held name outranks a later window's exact placement (window order
outranks pass order across windows), and the ids an earlier window placed are `used` under a later
window that resolves them. A name no window placed or held carries its most specific claim.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple

from .manual_import import EntryClaim, FileEpisodeMap
from .placement_types import EpisodeAssignment, EpisodeIndex, Placement, PlacementBatch, PlacementVerdict, TargetScope
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


# A file no window placed or held merges to its most specific claim: excluded only when every window excluded
# it, a refused duplicate over a file placed nowhere, and an extra (one only against its window's titles) under all.
_CLAIM_RANK: Mapping[PlacementVerdict, int] = {
    PlacementVerdict.EXTRA: 0,
    PlacementVerdict.FOREIGN: 1,
    PlacementVerdict.SKIPPED: 2,
    PlacementVerdict.DUPLICATE: 3,
}


def assign_across_windows(batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """Place a torrent's files under every window in order and merge the verdicts (see the module).

    One window is exactly one `assign_episode_ids` call over the same batch and scope. Under no
    windows at all every file is skipped.
    """

    names = batch.names
    decided: dict[str, WindowVerdict] = {}
    claims: dict[str, list[Placement]] = {name: [] for name in names}
    for index, window in enumerate(windows):
        placed_ids = frozenset(i for v in decided.values() for i in v.placement.ids)
        # An unscoped window can place onto any id, so every earlier placement is a seed there.
        seeds = placed_ids if window.unscoped else placed_ids & window.real_ids
        scope = replace(window, used=window.used | seeds)
        for placement in assign_episode_ids(batch, scope).placements:
            if placement.name in decided:
                continue
            if placement.verdict.placed:
                decided[placement.name] = WindowVerdict(placement, index)
            elif placement.verdict is PlacementVerdict.HELD:
                decided[placement.name] = WindowVerdict(placement, None)
            else:
                claims[placement.name].append(placement)
    return WindowedAssignment(tuple(decided.get(name) or _merged_claim(name, claims[name]) for name in names))


def _merged_claim(name: str, claims: Sequence[Placement]) -> WindowVerdict:
    """The most specific claim the windows made on a file none placed or held (`_CLAIM_RANK`)."""

    if not claims:
        return WindowVerdict(Placement(name, (), PlacementVerdict.SKIPPED), None)
    return WindowVerdict(max(claims, key=lambda claim: _CLAIM_RANK[claim.verdict]), None)


def windows_of(claims: Iterable[EntryClaim], indexes: Mapping[int, EpisodeIndex]) -> tuple[TargetScope, ...]:
    """One placement window per claim in order, over its series' index (`indexes` holds every claim's series)."""

    return tuple(
        TargetScope(list(claim.ordered_episode_ids), indexes[claim.series_id], names=claim.names) for claim in claims
    )


def place_leftover(seeded: FileEpisodeMap, batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """The names of `batch` the map does not cover, placed under `windows` with the mapped ids already used.

    The mapped ids ride each window as `used` narrowed to the window's own ids (every id under an
    unscoped window), the same rule the cross-window seeds follow, so one series' placements never
    close another series' count legs.
    """

    leftover = [name for name in batch.to_place if name not in seeded]
    mapped = frozenset(ep_id for ids in seeded.values() for ep_id in ids)
    narrowed = tuple(
        replace(window, used=window.used | (mapped if window.unscoped else mapped & window.real_ids))
        for window in windows
    )
    return assign_across_windows(PlacementBatch(leftover, batch.parsed), narrowed)
