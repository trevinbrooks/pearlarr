"""Grab-time placement: each url's files placed under the entry's scope, the planner's records, and the seeds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

from .coverage import coverage_string, episodes_from_ep_list
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, GroupVotes
from .manual_import import (
    NO_SIZES_BY_NAME,
    EntryClaim,
    EntryNames,
    FileEpisodeMap,
    GuardFacts,
    OwnGroup,
    PendingImport,
    normalized_leaf,
    sizes_by_episode,
    unambiguous_sizes,
)
from .placement_types import (
    EpisodeAssignment,
    EpisodeIndex,
    PlacementBatch,
    PlacementVerdict,
    TargetScope,
    TorrentListings,
)
from .seadex_types import EpisodeRecord, FlaggedUrl, GrabHold, ParsedFileInfo, SeadexDict, SeadexUrlItem, flagged_urls
from .window_placement import place_leftover, windows_of


class SeedFile(NamedTuple):
    """One listed video file, its size and its cached grab-time `/parse`, as the placement reads it."""

    basename: str
    """The raw SeaDex basename (normalized only where the placement keys the batch)."""

    size: int
    """The listed size, carried onto the records the planner compares by size."""

    parse: ParsedFileInfo | None
    """Sonarr's parse of the name, or None when no fresh parse record exists (an unknown parse keeps
    the count-based passes from placing, exactly as it does at import time)."""


class SeedScope(NamedTuple):
    """What one entry's files are placed against: its own index, the whole-series map, and its names."""

    al_id: int
    """The entry claiming the placement: a stored claim of its own is replaced by the fresh window."""

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
        unread map places nothing: a grab-time verdict is final, while an import poll is retried.
        """

        return bool(self.entry.by_id and self.series.id_by_key)

    def target(self) -> TargetScope:
        """The placement scope the grab shares with import time: the entry's ids, nothing used yet."""

        return TargetScope(list(self.entry.by_id), self.series, names=self.names)


@dataclass(frozen=True, slots=True)
class KnownTorrent:
    """What the run knows of one listed torrent beyond the entry: its stored record and the windows listing it."""

    record: PendingImport | None
    """The store-resident record the torrent accretes onto, None when the torrent is new."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A claim's series left out was not read, which places nothing."""

    listed: frozenset[int] | None
    """The ids of every window of the entry's series listing the torrent (`TargetScope.listed`), None when a
    listing entry's record or window could not be read: a window of specials is then held, any other placed as
    if listed nowhere."""

    def windows(self, al_id: int, own: TargetScope) -> tuple[TargetScope, ...] | None:
        """The windows the torrent is placed under, None when a claim's series is unread.

        One per stored claim in claim order, `own` replacing this entry's or appended last, each carrying the
        listing (an unread one as empty): exactly the windows the import poll runs once the claim is replaced.
        """

        claims = () if self.record is None else self.record.claims
        if any(claim.series_id not in self.indexes for claim in claims):
            return None
        stored = windows_of(claims, self.indexes)
        position = next((i for i, claim in enumerate(claims) if claim.al_id == al_id), None)
        windows = (*stored, own) if position is None else (*stored[:position], own, *stored[position + 1 :])
        return tuple(replace(window, listed=self.listed or frozenset()) for window in windows)


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

    listings: TorrentListings
    """Each hash to the windows of the series' entries listing it, None where one of them could not be read."""

    def known(self, seadex_dict: SeadexDict) -> KnownTorrents:
        """What the run knows of each url's torrent, keyed by url.

        A hash-less url is a new torrent nothing lists: placed by its numbers alone, and no record tracks its
        import.
        """

        return {
            url_item.url: KnownTorrent(
                None if url_item.infohash is None else self.stored.get(url_item.infohash),
                self.indexes,
                frozenset() if url_item.infohash is None else self.listings[url_item.infohash],
            )
            for rg_item in seadex_dict.values()
            for url_item in rg_item.urls.values()
        }


class UrlPlacement(NamedTuple):
    """One url's files placed at grab time: the verdicts, and the records the planner judges coverage by."""

    files: tuple[SeedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    assignment: EpisodeAssignment
    """This placement's verdicts, one per distinct normalized file name: the whole batch when the torrent
    is new, else the files the stored record's map does not cover."""

    records: tuple[EpisodeRecord, ...]
    """One `(season, episode, size)` per file and episode of the whole map inside the entry, in file order.
    A file nothing placed adds none. A url with no records makes the planner count its release group as
    covering every episode."""

    claimed_ids: frozenset[int]
    """The entry's episode ids the whole map covers (a stored placement on another series adds none)."""

    inputs_known: bool
    """Every parse came from Sonarr this run (`PlacementBatch.all_parses_known`) and every window the
    placement needed could be built. False means some files may have been held, so the title is re-checked
    next run."""

    hold: GrabHold | None
    """Why the url is not grabbed this run, None when it may be."""

    stored: PendingImport | None
    """The record the url was placed against, None when the torrent is new."""

    intended_sizes: Mapping[int, int]
    """Each mapped episode id to the listed size of the file placed on it.
    An id two files size differently is left out."""

    @property
    def sizes_by_name(self) -> dict[str, int]:
        """Normalized listing name -> its listed size (see `_sizes_by_name`)."""

        return _sizes_by_name(self.files)


def _sizes_by_name(files: Sequence[SeedFile]) -> dict[str, int]:
    """Normalized listing name -> its listed size, a name listed at two sizes left out."""

    return unambiguous_sizes((normalized_leaf(f.basename), f.size) for f in files)


def seed_batch(files: Sequence[SeedFile]) -> PlacementBatch:
    """One url's files as a placement batch keyed by NORMALIZED file name, as the on-disk names are at import time.

    NFC/NFD-safe: two listed files with one normalized name share its verdict (the first parse) and keep their sizes.
    """

    to_place = list(dict.fromkeys(normalized_leaf(f.basename) for f in files))
    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(normalized_leaf(f.basename), f.parse)
    return PlacementBatch(to_place, parsed)


def place_release(files: Sequence[SeedFile], scope: SeedScope, torrent: KnownTorrent) -> UrlPlacement:
    """Place one url's files ONCE, with the same `place_leftover` the import wait runs. Pure.

    Every window is built or none is: a claim's series unread, or the listing unread where it judges a window of
    specials, places nothing and holds a pack of specials (`GrabHold`), since import time would place it by number.
    """

    batch = seed_batch(files)
    seeded: dict[str, list[int]] = {} if torrent.record is None else torrent.record.seeded_map()
    own = scope.target()
    listed = torrent.listed
    # A read the listing's judgment needs holds the url only where it judges (`TargetScope.specials_listing`).
    judged = own.specials_window if listed is None else replace(own, listed=listed).specials_listing() is not None
    windows = None if judged and listed is None else torrent.windows(scope.al_id, own)
    if scope.can_place and windows is not None:
        assignment = place_leftover(seeded, batch, windows).merged
    else:
        assignment = EpisodeAssignment(())
    mapped = {**seeded, **assignment.assigned}
    records: list[EpisodeRecord] = []
    claimed: set[int] = set()
    for f in files:
        for ep_id in mapped.get(normalized_leaf(f.basename), []):
            # A stored id on another series is not this entry's. One inside the entry counts toward its coverage.
            episode = scope.entry.by_id.get(ep_id)
            if episode is None:
                continue
            claimed.add(ep_id)
            # An episode Sonarr sent without numbers fits no coverage key, so it adds no record.
            if episode.season_number is None or episode.episode_number is None:
                continue
            records.append(EpisodeRecord(season=episode.season_number, episode=episode.episode_number, size=f.size))
    # A new torrent under an unread map is judged coarsely, as before the placement existed. A stored one's
    # verdict would be final where the import poll retries, so its title is re-checked instead.
    known = windows is not None and (torrent.record is None or scope.can_place)
    # A url with no video file (subs, fonts) places nothing and waits on nothing.
    return UrlPlacement(
        files=tuple(files),
        assignment=assignment,
        records=tuple(records),
        claimed_ids=frozenset(claimed),
        inputs_known=not files or (batch.all_parses_known and known),
        hold=_hold_of(assignment, unread=judged and (listed is None or not batch.all_parses_known)) if files else None,
        stored=torrent.record,
        intended_sizes=sizes_by_episode(mapped, _sizes_by_name(files)),
    )


def _hold_of(assignment: EpisodeAssignment, *, unread: bool) -> GrabHold | None:
    """Why the url is not grabbed: a misnumbered verdict, else a read the listing's judgment needed that failed."""

    if any(p.verdict is PlacementVerdict.MISNUMBERED for p in assignment.placements):
        return GrabHold.MISNUMBERED
    return GrabHold.INPUT_UNREAD if unread else None


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
        """Write each url's placed records and hold onto its item, and the records' union onto its group."""

        for rg_item in seadex_dict.values():
            all_episodes: list[EpisodeRecord] = []
            for url, url_item in rg_item.urls.items():
                placement = self.by_url[url]
                url_item.episodes = list(placement.records)
                url_item.hold = placement.hold
                all_episodes.extend(url_item.episodes)
            rg_item.all_episodes = all_episodes

    def input_missing_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a url whose placement waits on a read that failed."""

        return tuple(
            group for group, item in seadex_dict.items() if not all(self.by_url[url].inputs_known for url in item.urls)
        )


class ClaimWindow(NamedTuple):
    """What one claim records about a torrent: episode ids, titles, preowned ids, and slice (empty when unscoped)."""

    ordered_episode_ids: tuple[int, ...] = ()
    names: EntryNames = EntryNames()
    preowned_episode_ids: tuple[int, ...] = ()
    slice_coverage: str | None = None


UNSCOPED = ClaimWindow()
"""The window of a claim whose import reads nothing but the infohash (a Radarr grab)."""


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

    def claim(self, window: ClaimWindow) -> EntryClaim:
        """The entry's claim over `window`, its clock blank until the record is stamped."""

        return EntryClaim(
            al_id=self.al_id,
            series_id=self.series_id,
            title=self.title,
            coverage=self.coverage,
            url=self.url,
            ordered_episode_ids=window.ordered_episode_ids,
            names=window.names,
            preowned_episode_ids=window.preowned_episode_ids,
            slice_coverage=window.slice_coverage,
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
    """The names this placement proved never the record's to import."""

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
            )
        record = (
            self.stored.with_placements(self.placements)
            .with_exclusions(self.excluded)
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
    return entry.claim(
        ClaimWindow(
            ordered_episode_ids=tuple(index.by_id),
            names=scope.names,
            preowned_episode_ids=preowned,
            slice_coverage=slice_coverage,
        )
    )


def build_pending_seed(release: SeedRelease, scope: SeedScope, entry: EntryFacts) -> PendingSeed:
    """Fold one placed release into the entry's seed on its torrent. Pure, never places.

    A placed file is seeded, an excluded one recorded as never the record's, and a held or skipped one
    left for import time, where the parses are re-read.
    """

    placed = release.placed
    return PendingSeed(
        facts=release.facts,
        placements=placed.assignment.assigned,
        excluded=tuple(placement.name for placement in placed.assignment.excluded),
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
