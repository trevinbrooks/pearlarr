"""A torrent's files placed under each title it's listed under, one window per title.

The windows run season windows first, then specials windows, the kept titles before the ones in `ignore_anilist_ids`
within each kind, and otherwise in AniList id order (`ordered_windows`), so a torrent places the same whichever title
sees it first and whether or not a pending import exists. Season windows go first because the sequence-over-parse
pass lets a specials window take a sequence whose spelled-out seasons and episodes lie outside it: a specials title
with a lower id than its season title would otherwise take the season's episodes. An ignored title goes after the
kept ones of the same kind because it never grabs: a file both could place is the kept title's to import, and the
ignored title gets only what no kept title places.

Every window places every file, not just the ones earlier windows left, because a numbered sequence cut short would
read as a new sequence and could land wrong. Then the windows merge in order:

- A file takes the first final verdict a window gives it, and its ids are used under the later windows. A file a
  window left for another TVDB order's episode (`OTHER_ORDER_OUTSIDE_WINDOW`) is placed by a later one only there.
- A file with no final verdict takes its highest-ranked provisional one (`PlacementVerdict.rank`).
- A file the special aliases point at under one window but its numbers place elsewhere is always `MISNUMBERED`.

The order of the season windows shows in what another TVDB order places (pass 16). A window that holds a key's
aired episode places the file there, and that placement is final against the order under every later window.

After the merge, the numbering step (`numberings`) places files in the gaps the windows left
(`WindowedPlacements.with_proposals`): a file named by number alone that every window left `NO_EPISODE_FOUND` or
`OUTSIDE_WINDOW`, that no window was involved in or parsed inside itself, is placed where its numbers fall when
every way the group could have numbered them agrees. The numbering step never moves a file a window gave a final
verdict, and a file it places counts as placed by the earliest window holding its ids.

A pending import imports only what its own titles' windows place, and takes on a hand import only when one of them
was involved in it (`TitleView.placements`).
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Self

from .numberings import NumberingProposals
from .pending_import import TitleClaim, TitleWindow
from .placement_types import (
    BatchPlacements,
    EpisodeIndex,
    Placement,
    PlacementBatch,
    PlacementTarget,
    PlacementVerdict,
    SeadexEvidence,
    TorrentListing,
    all_specials,
)
from .placer import NameFacts, place_batch


@dataclass(frozen=True, slots=True)
class WindowOrder:
    """The windows a torrent is placed under, in placement order, and which of them the pending import imports."""

    windows: tuple[TitleWindow, ...]
    """One per AniList id, in placement order (`ordered_windows`)."""

    imports: frozenset[int]
    """The positions whose files the pending import imports: the grabbing title's and every claimed title's."""

    grabbing_window_at: int
    """The position of the grabbing title's window."""

    other_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """Every episode id a window other than the grabbing title's holds."""

    def __post_init__(self) -> None:
        other_ids = frozenset(
            ep_id
            for at, window in enumerate(self.windows)
            if at != self.grabbing_window_at
            for ep_id in window.ordered_episode_ids
        )
        object.__setattr__(self, "other_ids", other_ids)


def ordered_windows(
    listing: TorrentListing,
    claims: Iterable[TitleClaim],
    grabbing_window: TitleWindow,
    indexes: Mapping[int, EpisodeIndex],
) -> WindowOrder | None:
    """Every window a torrent is placed under, one per AniList id, in placement order.

    The first three parameters are the three sources of windows, and `indexes` classifies each one. None when
    `indexes` lacks a window's series, since that window can be neither classified nor placed.

    Sorted as the module docstring says: season windows before specials windows, kept before ignored within a kind,
    then AniList id.

    A listed window with no ids is left out, since it would mean no window and admit any id. A claim's window with
    no ids still runs: `TitleClaim.admits` reads such a claim as covering any id. It sorts with the specials windows.

    A claim's window takes the ignored flag and the AniList episode count from its title's listed window, since the
    claim predates any ignore and doesn't keep the count. A claim whose title isn't listed has neither.
    """

    # A claim's window takes its title's slot from the listed one, since what the pending import imports is judged
    # against its claims, and `grabbing_window` takes it from both, since this grab replaces its claim. A claim whose
    # title the listing doesn't name keeps its window, and a title listed and never claimed gets its listed window.
    # The ignored flag and the episode count are the listing's for every window: facts about the title at the read,
    # which a claim doesn't hold. They're read off every listed window, so a title whose listed window came back
    # empty still gives them to its claim.
    chosen = {window.al_id: window for window in listing.windows if window.ordered_episode_ids}
    listed = {window.al_id: window for window in listing.windows}
    imported = set[int]()
    for claim in claims:
        listed_window = listed.get(claim.al_id)
        chosen[claim.al_id] = (
            claim.window
            if listed_window is None
            else replace(claim.window, ignored=listed_window.ignored, episode_count=listed_window.episode_count)
        )
        imported.add(claim.al_id)
    chosen[grabbing_window.al_id] = grabbing_window
    imported.add(grabbing_window.al_id)
    if any(window.series_id not in indexes for window in chosen.values()):
        return None
    windows = tuple(
        sorted(
            chosen.values(),
            key=lambda window: (
                all_specials(indexes[window.series_id], window.ordered_episode_ids),
                window.ignored,
                window.al_id,
            ),
        )
    )
    return WindowOrder(
        windows=windows,
        imports=frozenset(at for at, window in enumerate(windows) if window.al_id in imported),
        grabbing_window_at=windows.index(grabbing_window),
    )


@dataclass(frozen=True, slots=True)
class WindowFacts:
    """One window's target, with what the passes read off each name of the torrent under it (`PlacerResult.facts`)."""

    target: PlacementTarget
    names: Mapping[str, NameFacts]
    """Every name of the torrent, file map and gone names included."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "names", MappingProxyType(dict(self.names)))


@dataclass(frozen=True, slots=True)
class WindowedFile:
    """One file of the torrent over its windows."""

    merged: Placement
    """The file's verdict over every window, or the numbering step's: `NUMBERINGS_AGREE`, a verdict no window gives."""
    per_window: tuple[Placement, ...]
    """What the placer gave the file under each window, in window order. The numbering step changes none of them."""
    final_window_at: int | None
    """The position of the window that gave the file a final verdict, or, for a file the numbering step placed, of
    the window that holds its ids. A file whose aliases and numbers disagree keeps the window that gave its final
    verdict. None when no window gave a final verdict and the numbering step placed nothing."""

    @property
    def name(self) -> str:
        """The normalized file name."""

        return self.merged.name

    def involves(self, positions: Collection[int]) -> bool:
        """Whether a window at one of `positions` is involved in the file (`PlacementVerdict.involves`)."""

        return any(self.per_window[at].verdict.involves for at in positions)

    def is_another_titles(self, positions: Collection[int]) -> bool:
        """Whether the file is a title's outside the windows at `positions`.

        It is when a window outside them placed it, or when it's left for a hand import and no window among them was
        involved in it.
        """

        verdict = self.merged.verdict
        if verdict.placed:
            assert self.final_window_at is not None
            return self.final_window_at not in positions
        return verdict.left_for_hand_import and not self.involves(positions)


@dataclass(frozen=True, slots=True)
class TitleView:
    """What the windows made of a torrent's files, as one title and its pending import see it."""

    placements: BatchPlacements
    """The merged verdicts, with each file that is another title's set aside as `ANOTHER_TITLES`.

    A file is another title's when a window the pending import doesn't import placed it, or when it's left for a hand
    import (`PlacementVerdict.left_for_hand_import`) and no window the pending import imports was involved in that
    (`PlacementVerdict.involves`). The pending import neither imports nor hand-imports such a file: the title whose
    window judged it reports it under its own row."""
    misnumbered: bool
    """Whether a file the title's own window was involved in (`PlacementVerdict.involves`) ended up `MISNUMBERED`."""
    hand_import_verdict: PlacementVerdict | None
    """The hand-import verdict a file of the title's own window still carries after the merge, else None.

    The verdicts are `_HAND_IMPORT_VERDICTS`: contradicted by its sequence's titles, or read by another TVDB order
    outside the window. The window gave the file one and no window gave it a final verdict. `TITLE_REFUTED` comes
    first when both kinds are left."""
    another_titles: frozenset[str]
    """The files that are another title's for the title's own window alone (`placements`'s rule, with that window
    the only one imported). A file a window left `WAITING_ON_PARSE` isn't among them, since no window has read it
    yet."""


NOTHING_PLACED = TitleView(BatchPlacements(()), misnumbered=False, hand_import_verdict=None, another_titles=frozenset())
"""The view of a release whose windows placed nothing: no file left to place, or nothing to place it against."""


@dataclass(frozen=True, slots=True)
class WindowedPlacements:
    """A torrent placed under its windows: one `WindowedFile` per file, and what each window read off the names."""

    files: tuple[WindowedFile, ...]
    """One per name in `PlacementBatch.new_names`, batch order."""
    facts: tuple[WindowFacts, ...]
    """Each window's target, as `place_across_windows` was handed it, and what the passes read off every name under
    it, in window order."""

    def with_proposals(self, batch: PlacementBatch, proposals: NumberingProposals) -> Self:
        """The placements with each gap the numbering step proposed ids for placed on them, in batch order.

        A gap is a merged `NO_EPISODE_FOUND` or `OUTSIDE_WINDOW` file no window was involved in
        (`PlacementVerdict.involves`) and no window parsed inside itself (`NameFacts.parses_inside_window`): the passes
        never judged it by its own numbers, so a verdict on them is new, not a reversal. The proposed ids must be
        unused: on no merged placement, no name of the file map, and no gap placed earlier. The receiving window is the
        earliest whose target holds every id, and a name that carries, under it, the title of an episode outside them
        is left (the title check that blocks pairing in order).
        """

        if not proposals.ids:
            return self
        used = {ep_id for ids in batch.file_map.values() for ep_id in ids}
        used.update(ep_id for file in self.files for ep_id in file.merged.ids)
        files = list(self.files)
        for index, file in enumerate(self.files):
            ids = proposals.ids.get(file.name)
            if ids is None or not self._gap(file) or not used.isdisjoint(ids):
                continue
            position = next(
                (at for at, window in enumerate(self.facts) if window.target.real_ids.issuperset(ids)), None
            )
            if position is None:
                continue
            episode_by_title = self.facts[position].names[file.name].episode_by_title
            if episode_by_title is not None and episode_by_title not in ids:
                continue
            agreed = Placement(file.name, ids, PlacementVerdict.NUMBERINGS_AGREE)
            files[index] = replace(file, merged=agreed, final_window_at=position)
            used.update(ids)
        return replace(self, files=tuple(files))

    def _gap(self, file: WindowedFile) -> bool:
        """Whether the windows left the file a gap (`with_proposals`).

        Its merged verdict lets another order place it and is no hand import: nothing placed the file and nothing left
        it for a hand import. A specials file the aliases gave `ALIAS_OUTSIDE_WINDOW` merges to `NO_EPISODE_FOUND`
        too, which is why no window may be involved in it.
        """

        verdict = file.merged.verdict
        if not verdict.other_order_may_place or verdict.left_for_hand_import:
            return False
        if any(placement.verdict.involves for placement in file.per_window):
            return False
        return not any(window.names[file.name].parses_inside_window for window in self.facts)

    def seen_by(self, grabbing_window_at: int, imports: Collection[int]) -> TitleView:
        """What the windows made of the files, as the title whose window is at `grabbing_window_at` sees them.

        `imports` are the positions whose files the title's pending import imports (`WindowOrder.imports`).
        """

        another_titles = PlacementVerdict.ANOTHER_TITLES
        hand_import_verdicts = {
            file.merged.verdict
            for file in self.files
            if file.per_window[grabbing_window_at].verdict in _HAND_IMPORT_VERDICTS
            and file.merged.verdict in _HAND_IMPORT_VERDICTS
        }
        return TitleView(
            BatchPlacements(
                tuple(
                    Placement(file.name, (), another_titles) if file.is_another_titles(imports) else file.merged
                    for file in self.files
                )
            ),
            misnumbered=any(
                file.merged.verdict is PlacementVerdict.MISNUMBERED and file.involves((grabbing_window_at,))
                for file in self.files
            ),
            hand_import_verdict=max(hand_import_verdicts, key=lambda verdict: verdict.rank, default=None),
            another_titles=frozenset(file.name for file in self.files if file.is_another_titles((grabbing_window_at,))),
        )


_HAND_IMPORT_VERDICTS = frozenset({PlacementVerdict.TITLE_REFUTED, PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW})
"""Two of the three hand-import verdicts: contradicted by its sequence's titles, or read by another TVDB order
outside the window. Another window may still place such a file, one left for another order only on the episodes
that order reads it onto. `ORDER_ONLY`, the third, is left out."""


@dataclass(slots=True)
class _FileMerge:
    """One file's verdicts as the windows give them, in window order."""

    name: str
    per_window: list[Placement] = field(default_factory=list[Placement])
    """Every window's verdict, in window order."""
    weighed: list[Placement] = field(default_factory=list[Placement])
    """The verdicts the merge weighs: `per_window`, less a placement off the episodes in `other_order_episodes`."""
    final_window_at: int | None = None
    """The position of the first window that gave the file a final verdict."""
    other_order_episodes: tuple[int, ...] | None = None
    """The episodes the first `OTHER_ORDER_OUTSIDE_WINDOW` verdict read the file onto
    (`Placement.other_order_episodes`). A later window places it only there."""

    def take(self, placement: Placement) -> None:
        """Add the next window's verdict."""

        assert placement.name == self.name
        at = len(self.per_window)
        self.per_window.append(placement)
        verdict = placement.verdict
        if verdict.placed and self.other_order_episodes not in (None, placement.ids):
            return
        if verdict is PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW and self.other_order_episodes is None:
            self.other_order_episodes = placement.other_order_episodes
        self.weighed.append(placement)
        # A final verdict with no ids counts too, so no later window can place the file by its numbers.
        if self.final_window_at is None and verdict.final:
            self.final_window_at = at

    @property
    def placed_on(self) -> tuple[int, ...]:
        """The ids the window with the final verdict placed the file on: none before a final verdict, or with no ids."""

        return () if self.final_window_at is None else self.per_window[self.final_window_at].ids

    def over_windows(self) -> WindowedFile:
        """The file's verdict over every window.

        With no final verdict, every verdict the windows gave is provisional, so each has a rank
        (`PlacementVerdict.rank`).
        """

        if _alias_and_numbers_disagree(self.weighed):
            merged = Placement(self.name, (), PlacementVerdict.MISNUMBERED)
        elif self.final_window_at is not None:
            merged = self.per_window[self.final_window_at]
        elif not self.weighed:
            merged = Placement(self.name, (), PlacementVerdict.NO_EPISODE_FOUND)
        else:
            merged = max(self.weighed, key=lambda placement: placement.verdict.rank)
        return WindowedFile(merged, tuple(self.per_window), self.final_window_at)


def _alias_and_numbers_disagree(judged: Collection[Placement]) -> bool:
    """Whether the aliases gave the file a verdict under one window while another placed it on other ids by its numbers.

    A placement by numbers agrees only with a `SPECIAL_ALIAS` on the same ids, since an `ALIAS_OUTSIDE_WINDOW` has
    none.
    """

    from_alias = {placement.ids for placement in judged if placement.verdict.from_alias}
    if not from_alias:
        return False
    unaliased = {placement.ids for placement in judged if placement.verdict.placed and not placement.verdict.from_alias}
    return bool(unaliased - from_alias)


def targets_of(
    windows: Iterable[TitleWindow], indexes: Mapping[int, EpisodeIndex], listing: SeadexEvidence
) -> tuple[PlacementTarget, ...]:
    """One placement target per window, in order. `indexes` must hold every window's series.

    All targets share `listing`, but a size match only counts under a window on that episode's series.
    """

    return tuple(
        PlacementTarget(
            window.ordered_episode_ids,
            indexes[window.series_id],
            title_names=window.names,
            seadex=listing,
            episode_count=window.episode_count,
        )
        for window in windows
    )


def place_across_windows(batch: PlacementBatch, windows: Sequence[PlacementTarget]) -> WindowedPlacements:
    """The batch placed under each window in turn, verdicts merged.

    Each window is handed the files earlier windows placed (`PlacementTarget.after`): their ids are used where the
    window admits them. With no windows, every file is `NO_EPISODE_FOUND`.

    A misnumbered or contradicted file a window then left for another TVDB order's episode
    (`OTHER_ORDER_OUTSIDE_WINDOW`) is placed by a later window only on that episode. A placement of it anywhere else
    is dropped.
    """

    files = [_FileMerge(name) for name in batch.new_names]
    facts: list[WindowFacts] = []
    for window in windows:
        earlier = {file.name: ids for file in files if (ids := file.placed_on)}
        result = place_batch(batch, window.after(earlier))
        facts.append(WindowFacts(window, result.facts))
        for file, placement in zip(files, result.placements.placements, strict=True):
            file.take(placement)
    return WindowedPlacements(tuple(file.over_windows() for file in files), tuple(facts))
