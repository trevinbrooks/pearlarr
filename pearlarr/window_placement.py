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
- A file no window decides takes its strongest claim (`PlacementVerdict.claim_rank`).
- A file the special aliases claim under one window but its numbers place elsewhere is always `MISNUMBERED`.

The order of the season windows shows in what another TVDB order places (pass 16). A window that holds a key's
aired episode places the file there, which settles it against the order under every later window.

After the merge, the numberings reader (`numberings`) fills the gaps the windows left (`WindowedAssignment.filled`): a
file named by number alone that every window skipped or read as another slice's, that no window was party to or read
inside itself, is placed where its numbers fall when every way the group could have counted them agrees. The fill
never moves a file a window placed or refused, and a filled file counts as decided by the earliest window holding
its ids.

A record imports only what its own entries' windows place, and takes on a hand import only when one of them was party
to it (`EntryView.assignment`).
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .placer import NameFacts, assign_episode_ids


@dataclass(frozen=True, slots=True)
class WindowOrder:
    """The windows a torrent is placed under, in placement order, and which of them the record imports."""

    windows: tuple[EntryWindow, ...]
    """One per AniList id, in placement order (`ordered_windows`)."""

    imports: frozenset[int]
    """The positions whose files the record imports: the grabbing entry's and every claimed entry's."""

    own_at: int
    """The position of the grabbing entry's window."""

    other_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    """Every episode id a window other than the grabbing entry's holds."""

    def __post_init__(self) -> None:
        other_ids = frozenset(
            ep_id for at, window in enumerate(self.windows) if at != self.own_at for ep_id in window.ordered_episode_ids
        )
        object.__setattr__(self, "other_ids", other_ids)


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
        windows=windows,
        imports=frozenset(at for at, window in enumerate(windows) if window.al_id in imported),
        own_at=windows.index(own),
    )


@dataclass(frozen=True, slots=True)
class WindowFacts:
    """One window's scope, with what the passes read off each name of the torrent under it (`PlacerResult.facts`)."""

    scope: TargetScope
    names: Mapping[str, NameFacts]
    """Every name of the torrent, seeded and gone names included."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "names", MappingProxyType(dict(self.names)))


@dataclass(frozen=True, slots=True)
class WindowedFile:
    """One file of the torrent over its windows."""

    merged: Placement
    """The file's verdict over every window, or the fill's: `NUMBERINGS_AGREE`, a verdict no window gives."""
    per_window: tuple[Placement, ...]
    """What the placer gave the file under each window, in window order. A fill changes none of them."""
    decided_at: int | None
    """The position of the window that placed or refused the file, or, for a filled file, of the window that holds
    its ids. A file the aliases dispute keeps the window that decided it. None when no window decided the file and
    no fill placed it."""

    @property
    def name(self) -> str:
        """The normalized basename."""

        return self.merged.name

    def party(self, positions: Collection[int]) -> bool:
        """Whether a window at one of `positions` is party to the file (`PlacementVerdict.party`)."""

        return any(self.per_window[at].verdict.party for at in positions)

    def listed_elsewhere(self, positions: Collection[int]) -> bool:
        """Whether the file is a title's outside the windows at `positions`.

        It is when a window outside them placed it, or when it's left for a hand import and no window among them was
        party to it.
        """

        verdict = self.merged.verdict
        if verdict.placed:
            assert self.decided_at is not None
            return self.decided_at not in positions
        return verdict.hand_import and not self.party(positions)


@dataclass(frozen=True, slots=True)
class EntryView:
    """What the windows made of a torrent's files, as one entry and its record see it."""

    assignment: EpisodeAssignment
    """The merged verdicts, with each file that is another title's set aside as `LISTED_ELSEWHERE`.

    A file is another title's when a window the record doesn't import placed it, or when it's left for a hand import
    (`PlacementVerdict.hand_import`) and no window the record imports was party to that (`PlacementVerdict.party`).
    The record neither imports nor hand-imports such a file: the title whose window judged it reports it under its
    own row."""
    misnumbered: bool
    """Whether a file the entry's own window was party to (`PlacementVerdict.party`) ended up `MISNUMBERED`."""
    hand_claim: PlacementVerdict | None
    """The hand-import claim a file of the entry's own window still carries after the merge, else None.

    The claims are `_HAND_CLAIMS`: refuted by its title, or read by another TVDB order outside the window. The
    window made one for the file and no window placed or refused it. A title's refusal comes first when both kinds
    are left."""
    outside: frozenset[str]
    """The files that are another title's for the entry's own window alone (`assignment`'s rule, with that window
    the only one imported). A file a window held for a re-ask isn't among them, since no window has read it yet."""


NOTHING_PLACED = EntryView(EpisodeAssignment(()), misnumbered=False, hand_claim=None, outside=frozenset())
"""The view of a url whose windows placed nothing: no file left to place, or nothing to place it against."""


@dataclass(frozen=True, slots=True)
class WindowedAssignment:
    """A torrent placed under its windows: one record per file, and what each window read off the names."""

    files: tuple[WindowedFile, ...]
    """One per name in `PlacementBatch.names`, batch order."""
    facts: tuple[WindowFacts, ...]
    """Each window's scope, as `place_across_windows` was handed it, and what the passes read off every name under it,
    in window order."""

    def filled(self, batch: PlacementBatch, reading: NumberingReading) -> Self:
        """The assignment with each gap the reader proposed ids for filled, in batch order.

        A gap is a merged `SKIPPED` or `FOREIGN` file no window was party to (`PlacementVerdict.party`) and no
        window's reading resolved inside (`NameFacts.reads_inside`): the passes never judged it by its own numbers,
        so a verdict on them is new, not a reversal. The proposed ids must be free, held by no merged placement, no
        name of the stored map, and no earlier fill. The receiving window is the earliest whose scope holds every id,
        and a name titled under it as an episode outside them is left (the title veto the zips apply).
        """

        if not reading.ids:
            return self
        used = {ep_id for ids in batch.seeded.values() for ep_id in ids}
        used.update(ep_id for file in self.files for ep_id in file.merged.ids)
        files = list(self.files)
        for index, file in enumerate(self.files):
            ids = reading.ids.get(file.name)
            if ids is None or not self._gap(file) or not used.isdisjoint(ids):
                continue
            position = next((at for at, window in enumerate(self.facts) if window.scope.real_ids.issuperset(ids)), None)
            if position is None:
                continue
            titled_as = self.facts[position].names[file.name].titled_as
            if titled_as is not None and titled_as not in ids:
                continue
            agreed = Placement(file.name, ids, PlacementVerdict.NUMBERINGS_AGREE)
            files[index] = replace(file, merged=agreed, decided_at=position)
            used.update(ids)
        return replace(self, files=tuple(files))

    def _gap(self, file: WindowedFile) -> bool:
        """Whether the windows left the file a gap (`filled`).

        Its merged verdict is left open and no hand import: nothing placed the file and nothing settled it for a hand
        import. An alias-claimed specials file merges to `SKIPPED` too, which is why no window may be party to it.
        """

        verdict = file.merged.verdict
        if not verdict.left_open or verdict.hand_import:
            return False
        if any(placement.verdict.party for placement in file.per_window):
            return False
        return not any(window.names[file.name].reads_inside for window in self.facts)

    def seen_by(self, own_at: int, imports: Collection[int]) -> EntryView:
        """What the windows made of the files, as the entry whose window is at `own_at` sees them.

        `imports` are the positions whose files the entry's record imports (`WindowOrder.imports`).
        """

        elsewhere = PlacementVerdict.LISTED_ELSEWHERE
        hand_claims = {
            file.merged.verdict
            for file in self.files
            if file.per_window[own_at].verdict in _HAND_CLAIMS and file.merged.verdict in _HAND_CLAIMS
        }
        return EntryView(
            EpisodeAssignment(
                tuple(
                    Placement(file.name, (), elsewhere) if file.listed_elsewhere(imports) else file.merged
                    for file in self.files
                )
            ),
            misnumbered=any(
                file.merged.verdict is PlacementVerdict.MISNUMBERED and file.party((own_at,)) for file in self.files
            ),
            hand_claim=max(hand_claims, key=lambda verdict: verdict.claim_rank, default=None),
            outside=frozenset(file.name for file in self.files if file.listed_elsewhere((own_at,))),
        )


_HAND_CLAIMS = frozenset({PlacementVerdict.TITLE_REFUTED, PlacementVerdict.OTHER_ORDER_ELSEWHERE})
"""Two of the three hand-import claims: refuted by its run's titles, or read by another TVDB order outside the
window. Another window may still place such a file, one claimed for another order only on the episodes that order
reads it onto. `ORDER_ONLY`, the third, is left out."""


@dataclass(slots=True)
class _FileMerge:
    """One file's verdicts as the windows give them, in window order."""

    name: str
    per_window: list[Placement] = field(default_factory=list[Placement])
    """Every window's verdict, in window order."""
    weighed: list[Placement] = field(default_factory=list[Placement])
    """The verdicts the merge weighs: `per_window`, less a placement off the episodes in `read_onto`."""
    decided_at: int | None = None
    """The position of the first window that placed or refused the file."""
    read_onto: tuple[int, ...] | None = None
    """The episodes the first `OTHER_ORDER_ELSEWHERE` claim read the file onto (`Placement.read_onto`). A later window
    places it only there."""

    def take(self, placement: Placement) -> None:
        """Add the next window's verdict."""

        assert placement.name == self.name
        at = len(self.per_window)
        self.per_window.append(placement)
        verdict = placement.verdict
        if verdict.placed and self.read_onto not in (None, placement.ids):
            return
        if verdict is PlacementVerdict.OTHER_ORDER_ELSEWHERE and self.read_onto is None:
            self.read_onto = placement.read_onto
        self.weighed.append(placement)
        # A refusal decides the file too, so no later window can place it by its numbers.
        if self.decided_at is None and verdict.decides:
            self.decided_at = at

    @property
    def placed_on(self) -> tuple[int, ...]:
        """The ids the deciding window placed the file on: none while undecided, or when it was refused."""

        return () if self.decided_at is None else self.per_window[self.decided_at].ids

    def over_windows(self) -> WindowedFile:
        """The file's verdict over every window.

        With no deciding window, every verdict the windows gave is a claim, so each has a rank
        (`PlacementVerdict.claim_rank`).
        """

        if _disputed(self.weighed):
            merged = Placement(self.name, (), PlacementVerdict.MISNUMBERED)
        elif self.decided_at is not None:
            merged = self.per_window[self.decided_at]
        elif not self.weighed:
            merged = Placement(self.name, (), PlacementVerdict.SKIPPED)
        else:
            merged = max(self.weighed, key=lambda claim: claim.verdict.claim_rank)
        return WindowedFile(merged, tuple(self.per_window), self.decided_at)


def _disputed(judged: Collection[Placement]) -> bool:
    """Whether the aliases claimed the file under one window while another placed it on other ids by its numbers.

    A placement by numbers agrees only with an `ALTERNATE` on the same ids, since an `ALIASED_ELSEWHERE` has none.
    """

    by_alias = {placement.ids for placement in judged if placement.verdict.by_alias}
    if not by_alias:
        return False
    unaliased = {placement.ids for placement in judged if placement.verdict.placed and not placement.verdict.by_alias}
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

    files = [_FileMerge(name) for name in batch.names]
    facts: list[WindowFacts] = []
    for window in windows:
        earlier = {file.name: ids for file in files if (ids := file.placed_on)}
        result = assign_episode_ids(batch, window.after(earlier))
        facts.append(WindowFacts(window, result.facts))
        for file, placement in zip(files, result.assignment.placements, strict=True):
            file.take(placement)
    return WindowedAssignment(tuple(file.over_windows() for file in files), tuple(facts))
