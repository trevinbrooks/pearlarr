"""A torrent's files placed under several windows. A window is the `TargetScope` of one entry the torrent is listed on.

Every window judges the WHOLE torrent, exactly as a record with one window does, so a numbered run keeps
its shape under every window (the remainder re-read as a fresh run could place onto a window the whole
run never fits). The verdicts merge in window order: the first window that places or holds a file decides
it, so an earlier window's `HELD` outranks a later window's exact placement (window order beats pass
order). Ids an earlier window placed count as used under each later window that admits them. A file no
window placed or held takes the highest-ranked verdict the windows gave it (`_CLAIM_RANK`).
"""

from collections.abc import Iterable, Mapping, Sequence
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


# Highest wins. SKIPPED outranks FOREIGN and EXTRA, so a file skipped under any window is never excluded as another
# slice's or as an extra. DUPLICATE outranks SKIPPED (a refused duplicate beats a file placed nowhere).
# EXTRA depends on each window's titles: lowest.
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
        # Ids earlier windows placed count as used here: the ones this window admits, all of them when unscoped.
        scope = window.using(i for v in decided.values() for i in v.placement.ids)
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
    """One window per claim, in claim order, over its series' index (`indexes` holds every claim's series)."""

    return tuple(
        TargetScope(list(claim.ordered_episode_ids), indexes[claim.series_id], names=claim.names) for claim in claims
    )


def place_leftover(seeded: FileEpisodeMap, batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """The names of `batch` the map does not cover, placed under `windows` with the mapped ids already used.

    Each window marks used only the mapped ids it admits, as with ids an earlier window placed, so one
    series' placements never turn off another series' count-based passes.
    """

    leftover = [name for name in batch.to_place if name not in seeded]
    mapped = [ep_id for ids in seeded.values() for ep_id in ids]
    narrowed = tuple(window.using(mapped) for window in windows)
    return assign_across_windows(PlacementBatch(leftover, batch.parsed), narrowed)
