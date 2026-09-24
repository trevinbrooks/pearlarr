"""Grab-time placement: urls placed under the entry's scope, the records `attach_records` writes onto it, the seeds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NamedTuple

from .coverage import coverage_string, episodes_from_ep_list
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, trusted_groups
from .manual_import import EntryClaim, EntryNames, FileEpisodeMap, GuardFacts, OwnGroup, PendingImport, normalized_leaf
from .placement_types import EpisodeAssignment, EpisodeIndex, PlacementBatch, TargetScope
from .seadex_types import EpisodeRecord, FlaggedUrl, ParsedFileInfo, SeadexDict, SeadexUrlItem, flagged_urls
from .window_placement import place_leftover, windows_of


class SeedFile(NamedTuple):
    """One listed video file, its size and its cached grab-time `/parse`, as the placement reads it."""

    basename: str
    """The raw SeaDex basename (normalized only where the placement keys the batch)."""

    size: int
    """The listed size, carried onto the records the planner compares by size."""

    parse: ParsedFileInfo | None
    """Sonarr's parse of the name, or None when no fresh parse record exists (an unknown parse holds
    every count leg closed, exactly as it does at import time)."""


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
        """Whether the scope is real enough to place against: an entry index and a served series map.

        An empty entry index must never read as "no scope" (the unscoped arm places against the live
        map), and an unread map places nothing: a grab-time verdict is final where an import poll is retried.
        """

        return bool(self.entry.by_id and self.series.id_by_key)

    def target(self) -> TargetScope:
        """The placement scope the grab shares with import time: the entry's ids, nothing used yet."""

        return TargetScope(list(self.entry.by_id), self.series, names=self.names)


@dataclass(frozen=True, slots=True)
class ResidentScope:
    """A stored record a listed torrent accretes onto, with the indexes its claims' windows read."""

    record: PendingImport
    """The store-resident record on the torrent."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A claim's series left out was not read, which places nothing."""

    @property
    def can_place(self) -> bool:
        """Whether every stored claim's window can be built (its series index was read this run)."""

        return all(series_id in self.indexes for series_id in self.record.series_ids)

    def windows(self, al_id: int, own: TargetScope) -> tuple[TargetScope, ...]:
        """The claims' windows in claim order, `own` in place of the entry's stored claim, else appended last.

        A re-flag runs exactly the windows the import poll will run once the claim is replaced.
        """

        stored = windows_of(self.record.claims, self.indexes)
        position = next((i for i, claim in enumerate(self.record.claims) if claim.al_id == al_id), None)
        if position is None:
            return (*stored, own)
        return (*stored[:position], own, *stored[position + 1 :])


type ResidentScopes = Mapping[str, ResidentScope]
"""The stored records among an entry's urls, keyed by url as `SeadexReleaseGroupItem.urls` keys them."""

NO_RESIDENTS: ResidentScopes = MappingProxyType({})


def resident_scopes(
    seadex_dict: SeadexDict,
    stored: Mapping[str, PendingImport],
    indexes: Mapping[int, EpisodeIndex],
) -> ResidentScopes:
    """One `ResidentScope` per url whose torrent `stored` holds (keyed by infohash), by url."""

    return {
        url_item.url: ResidentScope(stored[url_item.infohash], indexes)
        for rg_item in seadex_dict.values()
        for url_item in rg_item.urls.values()
        if url_item.infohash is not None and url_item.infohash in stored
    }


class UrlPlacement(NamedTuple):
    """One url's files placed at grab time: the verdicts, and the records the planner judges coverage by."""

    files: tuple[SeedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    assignment: EpisodeAssignment
    """This placement's verdicts: the whole batch when the torrent is new, the stored map's leftover
    when it accretes onto a record, one per distinct normalized leaf."""

    records: tuple[EpisodeRecord, ...]
    """One `(season, episode, size)` per file and episode of the whole map inside the entry, in file order.
    A file nothing placed leaves none, so a url whose files place nowhere blankets the planner's coverage."""

    claimed_ids: frozenset[int]
    """The entry's episode ids the whole map covers (a stored placement onto another series is none)."""

    inputs_known: bool
    """Every parse came from Sonarr this run (`PlacementBatch.all_parses_known`) and every window the
    placement needed was readable. False means a run may have been held, so the title re-checks next run."""

    stored: PendingImport | None
    """The record the url was placed against, None when the torrent is new."""


def seed_batch(files: Sequence[SeedFile]) -> PlacementBatch:
    """One url's files as a placement batch keyed by NORMALIZED leaf, as the on-disk leaves are at import time.

    NFC/NFD-safe: two raw files sharing a leaf share its verdict (the first parse read) and keep their own sizes.
    """

    to_place = list(dict.fromkeys(normalized_leaf(f.basename) for f in files))
    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(normalized_leaf(f.basename), f.parse)
    return PlacementBatch(to_place, parsed)


def place_release(files: Sequence[SeedFile], scope: SeedScope, resident: ResidentScope | None) -> UrlPlacement:
    """Place one url's files ONCE, by the `place_leftover` the import wait runs. Pure.

    A stored record's leftover is placed under every claim's window, and only when each window's series
    was read: a grab-time verdict under an unread map would be final where the import poll retries.
    """

    batch = seed_batch(files)
    seeded: dict[str, list[int]] = {} if resident is None else resident.record.seeded_map()
    placeable = scope.can_place and (resident is None or resident.can_place)
    if placeable:
        windows = (scope.target(),) if resident is None else resident.windows(scope.al_id, scope.target())
        assignment = place_leftover(seeded, batch, windows).merged
    else:
        assignment = EpisodeAssignment(())
    mapped = {**seeded, **assignment.assigned}
    records: list[EpisodeRecord] = []
    claimed: set[int] = set()
    for f in files:
        for ep_id in mapped.get(normalized_leaf(f.basename), []):
            # A resident id on another series is not this entry's; one inside it counts for its coverage.
            episode = scope.entry.by_id.get(ep_id)
            if episode is None:
                continue
            claimed.add(ep_id)
            # An episode Sonarr sent unnumbered keys into no coverage, so it records nothing (a blanket).
            if episode.season_number is None or episode.episode_number is None:
                continue
            records.append(EpisodeRecord(season=episode.season_number, episode=episode.episode_number, size=f.size))
    inputs_known = batch.all_parses_known and (resident is None or placeable)
    return UrlPlacement(
        tuple(files),
        assignment,
        tuple(records),
        frozenset(claimed),
        inputs_known,
        None if resident is None else resident.record,
    )


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
        residents: ResidentScopes,
    ) -> "EntryPlacements":
        """Place each url's gathered files under one scope, a url with a stored record against it."""

        return cls(scope, {url: place_release(files, scope, residents.get(url)) for url, files in files_by_url.items()})

    def attach_records(self, seadex_dict: SeadexDict) -> None:
        """Write each url's placed records onto its item, and their union onto its group, for the planner."""

        for rg_item in seadex_dict.values():
            all_episodes: list[EpisodeRecord] = []
            for url, url_item in rg_item.urls.items():
                url_item.episodes = list(self.by_url[url].records)
                all_episodes.extend(url_item.episodes)
            rg_item.all_episodes = all_episodes

    def input_missing_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a url whose placement waits on a Sonarr read that failed."""

        return tuple(
            group for group, item in seadex_dict.items() if not all(self.by_url[url].inputs_known for url in item.urls)
        )


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
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingSeed:
    """One entry's contribution to a torrent's record. Only `record_at` builds the record, with the pipeline's stamp."""

    facts: TorrentFacts
    """The torrent's identity fields (a new record's; an accreted record keeps its own)."""

    placements: FileEpisodeMap
    """This placement's fresh map: the whole batch when the torrent is new, the stored map's leftover otherwise."""

    excluded: tuple[str, ...]
    """The names this placement proved never the record's to import."""

    claim: EntryClaim
    """This entry's fresh claim, its `claimed_at` blank until `record_at` stamps it."""

    stored: PendingImport | None
    """The record the torrent accretes onto, None when it is new."""

    @property
    def accreted(self) -> bool:
        """Whether the seed folds into a store-resident record."""

        return self.stored is not None

    def record_at(self, stamp: str, *, fresh: bool) -> PendingImport:
        """The record to persist, stamped: born at `stamp`, or accreted, every clock restarted when `fresh`."""

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
            )
        record = self.stored.with_placements(self.placements).with_exclusions(self.excluded).with_claim(claim)
        return record.restamped(stamp) if fresh else record


def build_entry_claim(release: SeedRelease, scope: SeedScope, entry: EntryFacts) -> EntryClaim:
    """The entry's claim on the release: the whole map's ids inside the entry, its window, slice, and preowned ids.

    Pure. The claim's clock is blank: `PendingSeed.record_at` stamps it.
    """

    claimed = release.placed.claimed_ids
    index = scope.entry
    # This claim's own slice of the entry, so records on sibling entries label distinctly: the episodes its
    # files claimed, else every episode it is verified against.
    slice_eps = [ep for ep in index.by_id.values() if not claimed or ep.id in claimed]
    # Targets that already hold a recommended file at grab time were never this torrent's to insert:
    # classify them against the claim's own trust slice (no sibling votes yet) so the wait's inserted
    # counts start at 0. A replaced claim keeps its first preowned ids (`PendingImport.with_claim`).
    grab_snapshot = EpisodeSnapshot(
        episodes=index,
        trusted=trusted_groups(entry.guards, release.own_group),
        owned_episode_sizes=entry.guards.owned_sizes,
    )
    preowned = tuple(
        ep_id
        for ep_id, status in grab_snapshot.statuses(sorted(claimed)).by_id.items()
        if status is EpisodeFileStatus.RECOMMENDED
    )
    return EntryClaim(
        al_id=entry.al_id,
        series_id=entry.series_id,
        title=entry.title,
        coverage=entry.coverage,
        url=entry.url,
        ordered_episode_ids=tuple(index.by_id),
        names=scope.names,
        preowned_episode_ids=preowned,
        slice_coverage=coverage_string(episodes_from_ep_list(slice_eps)) or None,
        claimed_at="",
        guards=entry.guards,
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
        ),
        placements={},
        excluded=(),
        claim=EntryClaim(
            al_id=entry.al_id,
            series_id=entry.series_id,
            title=entry.title,
            coverage=entry.coverage,
            url=entry.url,
            ordered_episode_ids=(),
            names=EntryNames(),
            preowned_episode_ids=(),
            slice_coverage=None,
            claimed_at="",
            guards=entry.guards,
        ),
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
