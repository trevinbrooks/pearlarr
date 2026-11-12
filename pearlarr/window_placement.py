"""A torrent's files placed under each entry it's listed on, one window per entry.

The windows run season windows first, then specials windows, the kept titles before the ones in `ignore_anilist_ids`
within each kind, and otherwise in AniList id order (`ordered_windows`), so a torrent places the same whichever entry
sees it first and whether or not a record exists. Season windows go first because the release-run pass lets a
specials window take a run keyed outside it: a specials title with a lower id than its season title would otherwise
take the season's episodes. An ignored title goes after the kept ones of the same kind because it never grabs: a file
both could place is the kept title's to import, and the ignored title gets only what no kept title places.

Every window places every file, not just the ones earlier windows left, because a numbered run cut short would read
as a new run and could land wrong. Then the windows merge in order:

- The first window to place or refuse a file decides it, and its ids are used under the later windows. A file a
  window claimed for another TVDB order's episode (`OTHER_ORDER_ELSEWHERE`) is placed by a later one only there.
- A file no window decides takes its strongest claim (`_CLAIM_RANK`).
- A file the special aliases claim under one window but its numbers place elsewhere is always `MISNUMBERED`.

The order of the season windows shows in what another TVDB order places (pass 16). A window that holds a key's
aired episode places the file there, which settles it against the order under every later window.

After the merge, the numberings reader (`numberings`) fills the gaps the windows left (`WindowedAssignment.filled`): a
file named by number alone that every window skipped or read as another slice's, that no window was party to or read
inside itself, is placed where its numbers fall when every way the group could have counted them agrees. The fill
never moves a file a window placed or refused, and a filled file counts as decided by the earliest window holding
its ids.

A record imports only what its own entries' windows place, and takes on a hand import only when one of them was party
to it (`WindowedAssignment.for_record`).
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Self

from .manual_import import EntryClaim, EntryWindow
from .numberings import NumberingReading
from .placement_types import (
    EpisodeAssignment,
    EpisodeIndex,
    ListingEvidence,
    Placement,
    PlacementBatch,
    PlacementVerdict,
    TargetScope,
    TorrentListing,
    all_specials,
)
from .placer import NameFacts, assign_episode_ids, name_facts


@dataclass(frozen=True, slots=True)
class WindowOrder:
    """The windows a torrent is placed under, in placement order, and which of them the record imports."""

    windows: tuple[EntryWindow, ...]
    """One per AniList id, in placement order (`ordered_windows`)."""

    imports: frozenset[int]
    """The positions whose files the record imports: the grabbing entry's and every claimed entry's."""

    def at(self, al_id: int) -> int:
        """The position of `al_id`'s window. Raises `ValueError` when the order holds none."""

        return [window.al_id for window in self.windows].index(al_id)

    def ids_outside(self, al_id: int) -> frozenset[int]:
        """Every episode id a window other than `al_id`'s holds."""

        return frozenset(
            ep_id for window in self.windows if window.al_id != al_id for ep_id in window.ordered_episode_ids
        )


def ordered_windows(
    listing: TorrentListing,
    claims: Iterable[EntryClaim],
    own: EntryWindow,
    indexes: Mapping[int, EpisodeIndex],
) -> WindowOrder | None:
    """Every window a torrent is placed under, one per AniList id, in placement order.

    The first three parameters are the three sources of windows, and `indexes` classifies each one. None when
    `indexes` lacks a window's series, since that window can be neither classified nor placed.

    Sorted as the module docstring says: season windows before specials windows, kept before ignored within a kind,
    then AniList id.

    A listed window with no ids is left out, since it would be unscoped and admit any id. A claim's window with no
    ids still runs: `EntryClaim.admits` reads such a claim as covering any id. It sorts with the specials windows.

    A claim's window takes the ignored flag and the AniList episode count from its entry's listed window, since the
    claim predates any ignore and doesn't keep the count. A claim whose entry isn't listed has neither.
    """

    # A claim's window takes its entry's slot from the listed one, since the record's import is judged against its
    # claims, and `own` takes it from both, since this grab replaces its claim. A claim whose entry the listing
    # doesn't name keeps its window, and an entry listed and never claimed gets its listed window. The ignored flag
    # and the episode count are the listing's for every window: facts about the entry at the read, which a claim
    # doesn't hold. They're read off every listed window, so an entry whose listed window came back empty still
    # gives them to its claim.
    chosen = {window.al_id: window for window in listing.entries if window.ordered_episode_ids}
    listed = {window.al_id: window for window in listing.entries}
    imported = set[int]()
    for claim in claims:
        entry = listed.get(claim.al_id)
        chosen[claim.al_id] = (
            claim.window
            if entry is None
            else replace(claim.window, ignored=entry.ignored, episode_count=entry.episode_count)
        )
        imported.add(claim.al_id)
    chosen[own.al_id] = own
    imported.add(own.al_id)
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
        windows=windows, imports=frozenset(at for at, window in enumerate(windows) if window.al_id in imported)
    )


@dataclass(frozen=True, slots=True)
class WindowFacts:
    """One window's scope, with what the passes read off each name of the torrent under it (`name_facts`)."""

    scope: TargetScope
    names: Mapping[str, NameFacts]
    """Every name of the torrent, seeded and gone names included."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "names", MappingProxyType(dict(self.names)))

    @classmethod
    def of(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """The scope with the passes' reading of every name of `batch` under it."""

        return cls(scope, name_facts(batch, scope))


@dataclass(frozen=True, slots=True)
class WindowedAssignment:
    """A torrent placed under its windows: the merged verdicts, each window's own, and which window decided."""

    merged: EpisodeAssignment
    """One verdict per name in `PlacementBatch.names`, batch order, whichever window placed it. A file `filled`
    placed carries `NUMBERINGS_AGREE`, a verdict no window gives."""
    windows: tuple[EpisodeAssignment, ...]
    """What the placer returned under each window, in window order. A fill changes none of them: a filled file still
    reads as the windows left it here."""
    decided_by: Mapping[str, int]
    """Name -> the position of the window that placed or refused it, or, for a filled file, of the window that holds
    its ids. A file no window decided and no fill placed is absent."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "decided_by", MappingProxyType(dict(self.decided_by)))

    def filled(self, batch: PlacementBatch, reading: NumberingReading, facts: Sequence[WindowFacts]) -> Self:
        """The assignment with each gap the reader proposed ids for filled, in batch order.

        A gap is a merged `SKIPPED` or `FOREIGN` file no window was party to (`_party`) and no window's reading
        resolved inside (`NameFacts.reads_inside`): the passes never judged it by its own numbers, so a verdict on
        them is new, not a reversal. The proposed ids must be free, held by no merged placement, no name of the
        stored map, and no earlier fill. The receiving window is the earliest whose scope holds every id, and a name
        titled under it as an episode outside them is left (the title veto the zips apply). `facts` are per window,
        in window order, as `windows` is.
        """

        used = {ep_id for ids in batch.seeded.values() for ep_id in ids}
        used.update(ep_id for placement in self.merged.placements for ep_id in placement.ids)
        placements = list(self.merged.placements)
        decided_by = dict(self.decided_by)
        for index, placement in enumerate(placements):
            ids = reading.ids.get(placement.name)
            if ids is None or not self._gap(index, facts) or not used.isdisjoint(ids):
                continue
            position = next((at for at, window in enumerate(facts) if window.scope.real_ids.issuperset(ids)), None)
            if position is None:
                continue
            titled_as = facts[position].names[placement.name].titled_as
            if titled_as is not None and titled_as not in ids:
                continue
            placements[index] = Placement(placement.name, ids, PlacementVerdict.NUMBERINGS_AGREE)
            decided_by[placement.name] = position
            used.update(ids)
        return replace(self, merged=EpisodeAssignment(tuple(placements)), decided_by=decided_by)

    def _gap(self, index: int, facts: Sequence[WindowFacts]) -> bool:
        """Whether the file at `index` of the batch is a gap the windows left (`filled`)."""

        merged = self.merged.placements[index]
        if merged.verdict not in _GAP:
            return False
        if any(_party(window.placements[index].verdict) for window in self.windows):
            return False
        return not any(window.names[merged.name].reads_inside for window in facts)

    def for_record(self, imports: Collection[int]) -> EpisodeAssignment:
        """The merged verdicts, with each file that is another title's set aside as `LISTED_ELSEWHERE`.

        A file is another title's when a window outside `imports` placed it, or when it's left for a hand import
        (`PlacementVerdict.hand_import`) and no window in `imports` was party to that (`_party`). The record neither
        imports nor hand-imports such a file: the title whose window judged it reports it under its own row.
        """

        elsewhere = PlacementVerdict.LISTED_ELSEWHERE
        return EpisodeAssignment(
            tuple(
                Placement(p.name, (), elsewhere) if self._listed_elsewhere(at, imports) else p
                for at, p in enumerate(self.merged.placements)
            )
        )

    def listed_elsewhere_under(self, own_at: int) -> frozenset[str]:
        """The files that are another title's for the window at `own_at` alone (`for_record`'s rule, one window).

        A file a window held for a re-ask isn't among them, since no window has read it yet.
        """

        return frozenset(p.name for at, p in enumerate(self.merged.placements) if self._listed_elsewhere(at, {own_at}))

    def _listed_elsewhere(self, index: int, imports: Collection[int]) -> bool:
        """Whether the file at `index` of the batch is a title's outside `imports`.

        Every window's placements are in batch order (`EpisodeAssignment.placements`), so one index names one file
        under each of them.
        """

        merged = self.merged.placements[index]
        if merged.verdict.placed:
            return self.decided_by[merged.name] not in imports
        if merged.verdict.hand_import:
            return not any(_party(self.windows[at].placements[index].verdict) for at in imports)
        return False

    def hand_claim_under(self, own_at: int) -> PlacementVerdict | None:
        """The hand-import claim a file of the window at `own_at` still carries after the merge, else None.

        The claims are `_HAND_CLAIMS`: refuted by its title, or read by another TVDB order outside the window. The
        window made one for the file and no window placed or refused it. A title's refusal comes first when both
        kinds are left.
        """

        own = {p.name for p in self.windows[own_at].placements if p.verdict in _HAND_CLAIMS}
        left = {p.verdict for p in self.merged.placements if p.name in own and p.verdict in _HAND_CLAIMS}
        return max(left, key=_CLAIM_RANK.__getitem__, default=None)

    def misnumbered_under(self, own_at: int) -> bool:
        """Whether a file the window at `own_at` was party to (`_party`) ended up `MISNUMBERED`."""

        own = {p.name for p in self.windows[own_at].placements if _party(p.verdict)}
        return self._left_as({PlacementVerdict.MISNUMBERED}, own)

    def _left_as(self, verdicts: Collection[PlacementVerdict], names: Collection[str]) -> bool:
        """Whether the merge left any of `names` with one of `verdicts`."""

        return any(p.verdict in verdicts and p.name in names for p in self.merged.placements)


# A file no window decided ends with its highest-ranked claim. EXTRA is lowest because it rests on one window's
# titles. SKIPPED beats FOREIGN and EXTRA, so a file some window couldn't place is never written off, and
# ALIASED_ELSEWHERE beats FOREIGN because the aliases name the file as this series' special. ORDER_ONLY beats
# SKIPPED, so a file one window's order reads as no episode of Sonarr's stays a hand import where another window
# only skipped it, and a window that proves it a duplicate still wins. OTHER_ORDER_ELSEWHERE and TITLE_REFUTED
# are highest, so a file whose numbers a window refused or refuted stays a hand import.
_CLAIM_RANK: Mapping[PlacementVerdict, int] = {
    PlacementVerdict.EXTRA: 0,
    PlacementVerdict.FOREIGN: 1,
    PlacementVerdict.ALIASED_ELSEWHERE: 2,
    PlacementVerdict.SKIPPED: 3,
    PlacementVerdict.ORDER_ONLY: 4,
    PlacementVerdict.DUPLICATE: 5,
    PlacementVerdict.OTHER_ORDER_ELSEWHERE: 6,
    PlacementVerdict.TITLE_REFUTED: 7,
}
_HAND_CLAIMS = frozenset({PlacementVerdict.TITLE_REFUTED, PlacementVerdict.OTHER_ORDER_ELSEWHERE})
"""The hand-import verdicts that are claims, not refusals: another window may still place the file, one claimed
for another order only on the episodes that order reads it onto."""
_BY_ALIAS = frozenset({PlacementVerdict.ALTERNATE, PlacementVerdict.ALIASED_ELSEWHERE})
"""The verdicts the special aliases give: placed on the aliased special, or claimed for the window that holds it."""
_GAP = frozenset({PlacementVerdict.SKIPPED, PlacementVerdict.FOREIGN})
"""The merged verdicts a fill may replace: nothing placed the file and nothing settled it for a hand import. An
alias-claimed specials file merges to `SKIPPED` too, which is why `filled` also asks whether a window was party."""


def _decides(verdict: PlacementVerdict) -> bool:
    """Whether a window that gives this verdict decides the file: it placed or refused it."""

    return verdict.placed or verdict.refused


def _party(verdict: PlacementVerdict) -> bool:
    """Whether a window giving this verdict is party to the file.

    It placed or refused the file, left it for a hand import, or claimed it by alias. A window that wrote the file
    off (another slice's, a duplicate, an extra, a skip) isn't.
    """

    return _decides(verdict) or verdict in _BY_ALIAS or verdict.hand_import


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
    windows: Iterable[EntryWindow], indexes: Mapping[int, EpisodeIndex], listing: ListingEvidence
) -> tuple[TargetScope, ...]:
    """One placement scope per window, in order. `indexes` must hold every window's series.

    All scopes share `listing`, but a size match only counts under a window on that episode's series.
    """

    return tuple(
        TargetScope(
            window.ordered_episode_ids,
            indexes[window.series_id],
            names=window.names,
            listing=listing,
            episode_count=window.episode_count,
        )
        for window in windows
    )


def place_across_windows(batch: PlacementBatch, windows: Sequence[TargetScope]) -> WindowedAssignment:
    """The batch placed under each window in turn, verdicts merged.

    Each window is handed the files earlier windows placed (`TargetScope.after`): their ids are used where the
    window admits them. With no windows, every file is skipped.

    A file a window refused and then claimed for another TVDB order's episode (`OTHER_ORDER_ELSEWHERE`) is placed
    by a later window only on that episode. A placement of it anywhere else is dropped.
    """

    names = batch.names
    placed: dict[str, tuple[int, ...]] = {}
    decided: dict[str, Placement] = {}
    decided_by: dict[str, int] = {}
    judged: dict[str, list[Placement]] = {name: [] for name in names}
    lifted: dict[str, tuple[int, ...]] = {}
    assignments: list[EpisodeAssignment] = []
    for position, window in enumerate(windows):
        assignment = assign_episode_ids(batch, window.after(placed))
        assignments.append(assignment)
        for placement in assignment.placements:
            verdict = placement.verdict
            if verdict.placed and lifted.get(placement.name, placement.ids) != placement.ids:
                continue
            if verdict is PlacementVerdict.OTHER_ORDER_ELSEWHERE:
                lifted.setdefault(placement.name, placement.read_onto)
            judged[placement.name].append(placement)
            # A refusal decides the file too, so no later window can place it by its numbers.
            if placement.name not in decided and _decides(verdict):
                decided[placement.name] = placement
                decided_by[placement.name] = position
                if placement.ids:
                    placed[placement.name] = placement.ids
    merged = EpisodeAssignment(tuple(_merged(name, judged[name], decided.get(name)) for name in names))
    return WindowedAssignment(merged, tuple(assignments), decided_by)
