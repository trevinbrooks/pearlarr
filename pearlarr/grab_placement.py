"""Grab-time placement: each url's files placed under the entry's scope, the planner's records, and the seeds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import NamedTuple

from .coverage import coverage_string, episodes_from_ep_list
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, GroupVotes
from .manual_import import (
    NO_SIZES_BY_NAME,
    EntryClaim,
    EntryNames,
    EntryWindow,
    FileEpisodeMap,
    GuardFacts,
    OwnGroup,
    PendingImport,
    joined_map,
    normalized_leaf,
    sizes_by_episode,
)
from .placement_types import (
    EMPTY_LISTING,
    NO_SEEDED,
    EpisodeAssignment,
    EpisodeIndex,
    ListingEvidence,
    ListingsRead,
    PlacementBatch,
    PlacementVerdict,
    SizeIdentities,
    TorrentListing,
)
from .seadex_types import (
    NO_ELSEWHERE,
    ElsewhereCounts,
    EpisodeKey,
    EpisodeRecord,
    FlaggedUrl,
    GrabHold,
    ParsedFileInfo,
    SeadexDict,
    SeadexUrlItem,
    flagged_urls,
    season_episode_key,
    unambiguous,
)
from .window_placement import WindowOrder, ordered_windows, place_across_windows, windows_of


class SeedFile(NamedTuple):
    """One listed video file, its size and its cached grab-time `/parse`, as the placement reads it."""

    basename: str
    """The raw SeaDex basename (normalized only where the placement keys the batch)."""

    size: int
    """The listed size, carried onto the records the planner compares by size."""

    parse: ParsedFileInfo | None
    """Sonarr's parse of the name, or None when no fresh parse record exists. An unknown parse keeps the count-based
    passes from placing, and holds the url while it has a file left to place (`GrabHold.INPUT_UNREAD`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedScope:
    """What one entry's files are placed against: its own index, the whole-series map, and its names."""

    al_id: int
    """The entry claiming the placement: a stored claim of its own is replaced by the fresh window."""

    series_id: int
    """The Sonarr series the entry's episodes belong to."""

    entry: EpisodeIndex
    """The entry's episodes: the resolved set, and what the slice and preowned reads cover."""

    series: EpisodeIndex
    """The WHOLE series (empty when the series list could not be read, which places nothing)."""

    names: EntryNames
    """The series and AniList titles the placement breaks ties by, persisted onto every seed."""

    @property
    def can_place(self) -> bool:
        """Whether there is an entry index and a served series map to place against.

        An empty entry index must not pass as "no scope" (which places against the whole series map), and an
        unread map places nothing: a grab-time verdict is final.
        """

        return bool(self.entry.by_id and self.series.id_by_key)

    @property
    def window(self) -> EntryWindow:
        """The entry's own window, built as a listing builds an entry's (`EntryWindow.of`)."""

        return EntryWindow.of(
            al_id=self.al_id, series_id=self.series_id, ep_list=self.entry.by_id.values(), names=self.names
        )


class TorrentEvidence(NamedTuple):
    """What this run read from SeaDex about one torrent: its listing and the series' size identities."""

    listing: TorrentListing | None
    """None if an entry listing the torrent couldn't be read, which places nothing: the windows of the entries
    listing it are unknown."""

    identities: SizeIdentities | None
    """None if a listing couldn't be read. The url is then held and the title re-checked next run."""


@dataclass(frozen=True, slots=True)
class KnownTorrent:
    """Everything the run knows about one torrent: its stored record, the series indexes, and the SeaDex evidence."""

    record: PendingImport | None
    """The store-resident record the torrent accretes onto, None when the torrent is new."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A window's series left out was not read, which places nothing."""

    evidence: TorrentEvidence
    """The torrent's listing and the series' size identities."""

    def listing_evidence(self, sizes: Mapping[str, int]) -> ListingEvidence:
        """The `ListingEvidence` every window of this torrent is placed with: this run's listing and size matches.

        `sizes` is each file's listed size by normalized name. No size matches when the identities weren't read.
        """

        listing, identities = self.evidence
        identified = {} if identities is None else {n: identities[s] for n, s in sizes.items() if s in identities}
        return ListingEvidence(EMPTY_LISTING if listing is None else listing, identified)

    def order(self, own: EntryWindow) -> WindowOrder | None:
        """The windows the torrent is placed under, None when the listing or a window's series is unread.

        Every window of this run's read, every stored claim's, and `own`, the grabbing entry's, in `ordered_windows`'
        order.
        """

        listing = self.evidence.listing
        if listing is None:
            return None
        return ordered_windows(listing, () if self.record is None else self.record.claims, own, self.indexes)


type KnownTorrents = Mapping[str, KnownTorrent]
"""One per url of the entry, keyed by url as `SeadexReleaseGroupItem.urls` keys them."""


def entry_hashes(seadex_dict: SeadexDict) -> frozenset[str]:
    """The infohashes among the entry's urls (a hash-less url has none)."""

    return frozenset(
        url_item.infohash
        for rg_item in seadex_dict.values()
        for url_item in rg_item.urls.values()
        if url_item.infohash is not None
    )


@dataclass(frozen=True, slots=True)
class TorrentReads:
    """The run's reads of the entry's torrents, each folded once: the store, the series indexes, the listings."""

    stored: Mapping[str, PendingImport]
    """The store-resident records among the entry's hashes."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A claim's series left out was not read, which places nothing."""

    listings: ListingsRead | None
    """The series' SeaDex listings, None when SeaDex was unreachable."""

    def known(self, seadex_dict: SeadexDict) -> KnownTorrents:
        """What the run knows about each url's torrent, keyed by url."""

        return {
            url_item.url: KnownTorrent(
                None if url_item.infohash is None else self.stored.get(url_item.infohash),
                self.indexes,
                self._evidence(url_item.infohash),
            )
            for rg_item in seadex_dict.values()
            for url_item in rg_item.urls.values()
        }

    def _evidence(self, infohash: str | None) -> TorrentEvidence:
        """The evidence for one url's torrent.

        A url with no infohash always gets `EMPTY_LISTING`, even when SeaDex is down, since no listing could match
        it anyway.
        """

        if self.listings is None:
            return TorrentEvidence(EMPTY_LISTING if infohash is None else None, None)
        listing = EMPTY_LISTING if infohash is None else self.listings.listing(infohash)
        return TorrentEvidence(listing, self.listings.identities)


@dataclass(frozen=True, slots=True, kw_only=True)
class UrlPlacement:
    """One url's files placed at grab time: the verdicts, and the records the planner judges coverage by."""

    files: tuple[SeedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    assignment: EpisodeAssignment
    """This placement's verdicts, one per distinct normalized file name: the whole batch when the torrent
    is new, else the files the stored record's map does not cover. A file that is another listed title's is
    `LISTED_ELSEWHERE` (`WindowedAssignment.for_record`)."""

    records: tuple[EpisodeRecord, ...]
    """One `(season, episode, size)` per file and episode of the whole map inside the entry, in file order.
    A file nothing placed adds none, and a url held for a hand import (`GrabHold.hand_import`) has none. A url with
    no records makes the planner count its release group as covering every episode."""

    claimed_ids: frozenset[int]
    """The entry's episode ids the whole map covers (a stored placement on another series adds none)."""

    inputs_known: bool
    """True when the placement had everything it needed this run: every window was built, the size identities were
    read, and, while a file is left to place, every parse came from Sonarr (`PlacementBatch.all_parses_known`). When
    False, a file may have been held or placed without its size, so the title gets re-checked next run."""

    hold: GrabHold | None
    """Why the url is not grabbed this run, None when it may be."""

    hold_counts: ElsewhereCounts
    """For `GrabHold.PLACED_ELSEWHERE`, how many of the files are another title's and how many nothing placed.
    `NO_ELSEWHERE` for any other url."""

    stored: PendingImport | None
    """The record the url was placed against, None when the torrent is new."""

    intended_sizes: Mapping[int, int]
    """Each mapped episode id to the listed size of the file placed on it.
    An id two files size differently is left out."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "intended_sizes", MappingProxyType(dict(self.intended_sizes)))

    @property
    def sizes_by_name(self) -> dict[str, int]:
        """Normalized listing name -> its listed size (see `_sizes_by_name`)."""

        return _sizes_by_name(self.files)


def _sizes_by_name(files: Sequence[SeedFile]) -> dict[str, int]:
    """Normalized listing name -> its listed size, a name listed at two sizes left out."""

    return unambiguous((normalized_leaf(f.basename), f.size) for f in files)


def seed_batch(files: Sequence[SeedFile], seeded: FileEpisodeMap = NO_SEEDED) -> PlacementBatch:
    """One url's files as a placement batch keyed by NORMALIZED file name, as the on-disk names are at import time.

    NFC/NFD-safe: two listed files with one normalized name share its verdict (the first parse) and keep their sizes.
    `seeded` is the stored record's map (`PendingImport.file_episode_map`), whose names are never placed.
    """

    to_place = list(dict.fromkeys(normalized_leaf(f.basename) for f in files))
    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(normalized_leaf(f.basename), f.parse)
    return PlacementBatch(to_place, parsed, seeded)


def place_release(files: Sequence[SeedFile], scope: SeedScope, torrent: KnownTorrent) -> UrlPlacement:
    """Place one url's files under the torrent's windows (`KnownTorrent.order`), with no I/O.

    Nothing is placed when the listing or a window's series wasn't read. A file another title's window placed, or
    left for a hand import with no window the record imports party to it, is set aside as that title's. The url is
    held (`GrabHold`) when a file of the entry's own window is misnumbered, when a read the placement needs failed,
    or when nothing of it lands on the entry and either its titles refute its numbers or some of it is another
    title's.
    """

    seeded: FileEpisodeMap = NO_SEEDED if torrent.record is None else torrent.record.file_episode_map
    batch = seed_batch(files, seeded=seeded)
    sizes = _sizes_by_name(files)
    evidence = torrent.listing_evidence(sizes)
    identities_unread = torrent.evidence.identities is None
    order = torrent.order(scope.window)
    # Every video file's parse feeds the checks on the files being placed, the seeded ones' included, and nothing
    # places a file after the grab, so an unknown parse holds the url while it has a file left to place. A url the
    # record maps whole needs no parse.
    parse_unread = bool(batch.names) and not batch.all_parses_known
    if scope.can_place and order is not None:
        own_at = order.at(scope.al_id)
        placed = place_across_windows(batch, windows_of(order.windows, torrent.indexes, evidence))
        assignment = placed.for_record(order.imports)
        misnumbered, refuted = placed.misnumbered_under(own_at), placed.refuted_under(own_at)
        outside = placed.listed_elsewhere_under(own_at)
    else:
        assignment, misnumbered, refuted, outside = EpisodeAssignment(()), False, False, frozenset[str]()
    mapped = joined_map(seeded, assignment.assigned)
    records: list[EpisodeRecord] = []
    claimed: set[int] = set()
    for f in files:
        for ep_id in mapped.get(normalized_leaf(f.basename), ()):
            # A stored id on another series is not this entry's. One inside the entry counts toward its coverage.
            episode = scope.entry.by_id.get(ep_id)
            if episode is None:
                continue
            claimed.add(ep_id)
            # An episode Sonarr sent without numbers fits no coverage key, so it adds no record.
            if episode.season_number is None or episode.episode_number is None:
                continue
            records.append(EpisodeRecord(season=episode.season_number, episode=episode.episode_number, size=f.size))
    # With every window's series read, a scope that can't place (`SeedScope.can_place`) still places nothing. A new
    # torrent is then judged by group and size. A stored one's claim joins with nothing placed, a grab-time verdict
    # that would be final, so its title is re-checked.
    known = order is not None and (torrent.record is None or scope.can_place)
    # An order that couldn't be built is a read that failed, since the url's windows are unknown.
    unread = order is None or identities_unread or parse_unread
    # With nothing of the whole map on the entry, a file another title's window placed or left for a hand import is
    # that title's, and so is a mapped file on an episode another title's window holds. A mapped file on an episode
    # no listed title's window holds is nobody's and doesn't hold the url. Judged on the whole map, so a sibling's
    # stored files count as its fresh ones would.
    other_ids = frozenset[int]() if order is None else order.ids_outside(scope.al_id)
    mapped_elsewhere = frozenset(name for name in batch.to_place if not other_ids.isdisjoint(mapped.get(name, ())))
    elsewhere = frozenset[str]() if claimed else outside | mapped_elsewhere
    # A url with no video file (subs, fonts) places nothing and waits on nothing.
    hold = (
        _hold_of(misnumbered=misnumbered, unread=unread, refuted=refuted and not claimed, elsewhere=bool(elsewhere))
        if files
        else None
    )
    return UrlPlacement(
        files=tuple(files),
        assignment=assignment,
        # A pack left for a hand import adds no records: the planner judges it by what its window holds.
        records=() if hold is not None and hold.hand_import else tuple(records),
        claimed_ids=frozenset(claimed),
        inputs_known=not files or (not parse_unread and known and not identities_unread),
        hold=hold,
        hold_counts=_elsewhere_counts(elsewhere, assignment) if hold is GrabHold.PLACED_ELSEWHERE else NO_ELSEWHERE,
        stored=torrent.record,
        intended_sizes=sizes_by_episode(mapped, sizes),
    )


def _hold_of(*, misnumbered: bool, unread: bool, refuted: bool, elsewhere: bool) -> GrabHold | None:
    """Why the url isn't grabbed: the first of these that holds, in this order.

    Four flags, since their order is the rule. `misnumbered`: a file of the entry's own window is misnumbered
    (`WindowedAssignment.misnumbered_under`). `unread`: a read the placement needs failed. `refuted`: the entry's
    own window refuted a file no window placed, and nothing of the whole map lands on the entry. It outranks
    `elsewhere`, since a file the entry's own window refuted is its own hand import, and saying it's another title's
    would be wrong. `elsewhere`: nothing of the whole map lands on the entry, and some file is another title's.
    """

    if misnumbered:
        return GrabHold.MISNUMBERED
    if unread:
        return GrabHold.INPUT_UNREAD
    if refuted:
        return GrabHold.TITLE_REFUTED
    return GrabHold.PLACED_ELSEWHERE if elsewhere else None


def _elsewhere_counts(elsewhere: frozenset[str], assignment: EpisodeAssignment) -> ElsewhereCounts:
    """How a url whose files are another title's splits: `elsewhere`, and the files nothing placed outside it.

    A file another title's window left for a hand import can be skipped as well as among `elsewhere` (a claimed
    title was party to it), so it counts once.
    """

    unplaced = sum(name not in elsewhere for name in assignment.skipped)
    return ElsewhereCounts(elsewhere=len(elsewhere), unplaced=unplaced)


class IdentityOverride(NamedTuple):
    """A file placed by size on an episode its name doesn't point at."""

    name: str
    """The normalized file name."""

    read_as: str
    """What the name reads as, for the warning: its own keys, else Sonarr's matched episodes, else its absolute
    numbers."""

    named: EpisodeKey
    """The episode its size matched, where it was placed."""


def identity_overrides(placed: UrlPlacement, series: EpisodeIndex) -> tuple[IdentityOverride, ...]:
    """One per file placed by size over its name (`PlacementVerdict.OVERRIDDEN`)."""

    parses = seed_batch(placed.files).parsed
    overrides: list[IdentityOverride] = []
    for placement in placed.assignment.placements:
        if placement.verdict is not PlacementVerdict.OVERRIDDEN:
            continue
        if (episode := series.by_id.get(placement.ids[0])) is not None:
            named = season_episode_key(episode.season_number, episode.episode_number)
            overrides.append(IdentityOverride(placement.name, _read_as(parses.get(placement.name)), named))
    return tuple(overrides)


def _read_as(info: ParsedFileInfo | None) -> str:
    """How the warning shows a parse: its own keys, else Sonarr's matched episodes, else its absolute numbers."""

    if info is None:
        return "nothing"
    if info.season_number is not None and info.episode_numbers:
        keys = [EpisodeKey(info.season_number, episode) for episode in info.episode_numbers]
    else:
        keys = [EpisodeKey(matched.season_number, matched.episode_number) for matched in info.matched_episodes]
    labels = [key.label for key in keys] or [f"absolute {number}" for number in info.absolute_episode_numbers]
    return ", ".join(labels) or "nothing"


class EntryPlacements(NamedTuple):
    """Every url of one entry placed, keyed as `SeadexReleaseGroupItem.urls` keys them."""

    scope: SeedScope
    """The scope every url was placed under (the seeds persist its names)."""

    by_url: Mapping[str, UrlPlacement]
    """One placement per url, every url present (an empty one for a url with no video file)."""

    @classmethod
    def place(
        cls,
        scope: SeedScope,
        files_by_url: Mapping[str, Sequence[SeedFile]],
        torrents: KnownTorrents,
    ) -> "EntryPlacements":
        """Place each url's files under the entry's scope, against what the run knows of its torrent."""

        return cls(scope, {url: place_release(files, scope, torrents[url]) for url, files in files_by_url.items()})

    def attach_placements(self, seadex_dict: SeadexDict) -> None:
        """Write each url's records, hold, and hold counts onto its item, and the records' union onto its group."""

        for rg_item in seadex_dict.values():
            all_episodes: list[EpisodeRecord] = []
            for url, url_item in rg_item.urls.items():
                placement = self.by_url[url]
                url_item.episodes = list(placement.records)
                url_item.hold = placement.hold
                url_item.hold_counts = placement.hold_counts
                all_episodes.extend(url_item.episodes)
            rg_item.all_episodes = all_episodes

    def input_missing_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a url whose placement waits on a read that failed."""

        return tuple(
            group for group, item in seadex_dict.items() if not all(self.by_url[url].inputs_known for url in item.urls)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimTerms:
    """What one claim records about a torrent: its window's ids and titles, its preowned ids, and its slice.

    All empty when unscoped.
    """

    ordered_episode_ids: tuple[int, ...] = ()
    names: EntryNames = field(default_factory=EntryNames)
    preowned_episode_ids: tuple[int, ...] = ()
    slice_coverage: str | None = None


UNSCOPED = ClaimTerms()
"""The terms of a claim whose import reads nothing but the infohash (a Radarr grab)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryFacts:
    """What an entry is, independent of any torrent: the fields every claim it makes carries."""

    al_id: int
    """The AniList entry id, the claim's key."""
    series_id: int
    """The Sonarr series id the entry's files belong to (0 for a Radarr movie)."""
    title: str
    """The AniList display title (logging only)."""
    coverage: str | None
    """The entry's season/episode coverage at grab time (logging only)."""
    url: str | None
    """The SeaDex entry URL at grab time (logging only)."""
    guards: GuardFacts
    """The plan's overwrite-guard evidence, copied onto every claim unchanged (see `GuardFacts`)."""

    def claim(self, terms: ClaimTerms) -> EntryClaim:
        """The entry's claim on `terms`, its clock blank until the record is stamped."""

        return EntryClaim(
            al_id=self.al_id,
            series_id=self.series_id,
            title=self.title,
            coverage=self.coverage,
            url=self.url,
            ordered_episode_ids=terms.ordered_episode_ids,
            names=terms.names,
            preowned_episode_ids=terms.preowned_episode_ids,
            slice_coverage=terms.slice_coverage,
            claimed_at="",
            guards=self.guards,
        )


class TorrentFacts(NamedTuple):
    """What a listed torrent is, independent of any entry: the record's identity fields."""

    infohash: str
    """The qBittorrent tracking key, lowercase."""
    release_group: str
    """The SeaDex release group (authoritative)."""
    is_dual_audio: bool
    """The listing's dual-audio flag."""
    seadex_files: tuple[str, ...]
    """The listing's video file names, the import's progress denominator."""
    release_sizes: tuple[int, ...]
    """The listing's file sizes, the trust policy's own-group vote."""
    sizes_by_name: Mapping[str, int]
    """Normalized listing name -> its listed size, the file each placed episode should hold (a name listed at two
    sizes is left out)."""


class SeedRelease(NamedTuple):
    """One flagged release's seed inputs: the url record and its placement (I/O done)."""

    release_group: str
    """The SeaDex release group (authoritative)."""

    url_item: SeadexUrlItem
    """The flagged url record whole: the fold reads its sizes and dual-audio flag."""

    infohash: str
    """The qBittorrent tracking key (the url item's hash, already narrowed to `str`)."""

    placed: UrlPlacement
    """The url's files placed at grab time (`place_release`)."""

    @property
    def own_group(self) -> OwnGroup:
        """The release's group at its listing's sizes, the trust policy's last vote."""

        return OwnGroup(self.release_group, tuple(self.url_item.size))

    @property
    def facts(self) -> TorrentFacts:
        """The torrent's identity fields as the record persists them."""

        return TorrentFacts(
            self.infohash,
            self.release_group,
            self.url_item.is_dual_audio,
            tuple(f.basename for f in self.placed.files),
            self.own_group.sizes,
            self.placed.sizes_by_name,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingSeed:
    """One entry's contribution to a torrent's record. Only `record_at` builds the record, with the pipeline's stamp."""

    facts: TorrentFacts
    """The torrent's identity fields for a new record. A stored record keeps its own."""

    placements: FileEpisodeMap
    """This placement's fresh map: the whole batch when the torrent is new, else the files the stored map lacks."""

    excluded: tuple[str, ...]
    """The names this placement proved aren't the record's to import under its current claims."""

    hand_import_files: tuple[str, ...]
    """The names this placement left for a hand import (`EpisodeAssignment.hand_import_files`), which keep the
    record pending once the placed files are in."""

    claim: EntryClaim
    """This entry's fresh claim, its `claimed_at` blank until `record_at` stamps it."""

    stored: PendingImport | None
    """The stored record this seed joins, None when the torrent is new."""

    @property
    def accreted(self) -> bool:
        """Whether the seed joins a stored record."""

        return self.stored is not None

    def record_at(self, stamp: str, *, fresh: bool) -> PendingImport:
        """The record to persist: new at `stamp`, or the stored one updated, with every clock reset when `fresh`."""

        claim = replace(self.claim, claimed_at=stamp)
        if self.stored is None:
            return PendingImport(
                infohash=self.facts.infohash,
                release_group=self.facts.release_group,
                is_dual_audio=self.facts.is_dual_audio,
                seadex_files=self.facts.seadex_files,
                added_at=stamp,
                file_episode_map=self.placements,
                claims=(claim,),
                excluded_files=self.excluded,
                release_sizes=self.facts.release_sizes,
                sizes_by_name=self.facts.sizes_by_name,
                hand_import_files=self.hand_import_files,
            )
        # The hand-import files join the stored ones rather than replace them: a grab whose scope can't place
        # carries none, and would otherwise clear an earlier grab's.
        record = (
            self.stored.with_placements(self.placements)
            .with_exclusions(self.excluded)
            .with_hand_import_files(self.hand_import_files)
            .with_claim(claim)
            .with_sizes_by_name(self.facts.sizes_by_name)
        )
        if not fresh:
            return record
        # A fresh add starts every clock: the birth and each claim's.
        return replace(record, added_at=stamp, claims=tuple(replace(c, claimed_at=stamp) for c in record.claims))


def build_entry_claim(release: SeedRelease, scope: SeedScope, entry: EntryFacts) -> EntryClaim:
    """The entry's claim on the release: its window, slice, and preowned ids. Pure, its clock blank."""

    claimed = release.placed.claimed_ids
    index = scope.entry
    # This claim's own slice of the entry, so records on sibling entries label distinctly: the episodes its
    # files claimed, else every episode it is verified against.
    slice_eps = [ep for ep in index.by_id.values() if not claimed or ep.id in claimed]
    # Targets that already hold a recommended file at grab time were never this torrent's to insert:
    # classify them against the claim's own trust slice (no sibling votes yet) so the wait's inserted
    # counts start at 0. A replaced claim keeps its first preowned ids (`PendingImport.with_claim`).
    grab_snapshot = EpisodeSnapshot.guarded(index, entry.guards, GroupVotes(release.own_group))
    preowned = tuple(
        ep_id
        for ep_id, status in grab_snapshot.statuses(sorted(claimed), release.placed.intended_sizes).by_id.items()
        if status is EpisodeFileStatus.RECOMMENDED
    )
    slice_coverage = coverage_string(episodes_from_ep_list(slice_eps)) or None
    window = scope.window
    return entry.claim(
        ClaimTerms(
            ordered_episode_ids=window.ordered_episode_ids,
            names=window.names,
            preowned_episode_ids=preowned,
            slice_coverage=slice_coverage,
        )
    )


def build_pending_seed(release: SeedRelease, scope: SeedScope, entry: EntryFacts) -> PendingSeed:
    """Fold one placed release into the entry's seed on its torrent. Pure, never places.

    A placed file is seeded, an excluded one recorded as not the record's under its current claims, and one left for
    a hand import recorded so the import can report it. A plain skip is recorded nowhere: the import warns of it as a
    file on disk the map lacks.
    """

    placed = release.placed
    return PendingSeed(
        facts=release.facts,
        placements=placed.assignment.assigned,
        excluded=tuple(placement.name for placement in placed.assignment.excluded),
        hand_import_files=placed.assignment.hand_import_files,
        claim=build_entry_claim(release, scope, entry),
        stored=placed.stored,
    )


def build_unscoped_seed(flagged: FlaggedUrl, entry: EntryFacts, stored: PendingImport | None) -> PendingSeed:
    """The seed of a torrent whose import reads nothing but its infohash (a Radarr grab): an id-less claim."""

    return PendingSeed(
        facts=TorrentFacts(
            infohash=flagged.infohash,
            release_group=flagged.group,
            is_dual_audio=flagged.item.is_dual_audio,
            seadex_files=(),
            release_sizes=(),
            sizes_by_name=NO_SIZES_BY_NAME,
        ),
        placements={},
        excluded=(),
        hand_import_files=(),
        claim=entry.claim(UNSCOPED),
        stored=stored,
    )


def build_pending_seeds(
    seadex_dict: SeadexDict,
    placed: EntryPlacements,
    entry: EntryFacts,
) -> dict[str, PendingSeed]:
    """Fold every flagged url carrying an infohash and a video file into its seed, keyed by infohash. Pure."""

    seeds: dict[str, PendingSeed] = {}
    for flagged in flagged_urls(seadex_dict):
        placement = placed.by_url[flagged.url]
        if not placement.files:
            continue
        release = SeedRelease(flagged.group, flagged.item, flagged.infohash, placement)
        seeds[flagged.infohash] = build_pending_seed(release, placed.scope, entry)
    return seeds
