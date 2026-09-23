"""Grab-time placement: urls placed under the entry's scope, the records `attach_records` writes onto it, the seeds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from .coverage import coverage_string, episodes_from_ep_list
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, trusted_groups
from .manual_import import EntryNames, GuardFacts, PendingImport, normalized_leaf
from .placement_types import EpisodeAssignment, EpisodeIndex, PlacementBatch, TargetScope
from .placer import assign_episode_ids
from .seadex_types import EpisodeRecord, ParsedFileInfo, SeadexDict, SeadexUrlItem, flagged_urls


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


class UrlPlacement(NamedTuple):
    """One url's files placed at grab time: the verdicts, and the records the planner judges coverage by."""

    files: tuple[SeedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    assignment: EpisodeAssignment
    """The placement verdicts, one per distinct normalized leaf."""

    records: tuple[EpisodeRecord, ...]
    """One `(season, episode, size)` per placed file and episode, in file order. A file nothing placed
    leaves none, so a url whose files place nowhere blankets the planner's coverage."""

    parses_known: bool
    """Every parse came from Sonarr this run (`PlacementBatch.all_parses_known`). False means a request
    failed and a run may be held, so the title is re-checked next run."""


def place_release(files: Sequence[SeedFile], scope: SeedScope) -> UrlPlacement:
    """Place one url's files by the same `assign_episode_ids` the import wait runs.

    Pure. The batch is keyed by NORMALIZED leaf so it matches the on-disk leaves at import time
    (NFC/NFD-safe): two raw files sharing a leaf share its verdict and keep their own sizes.
    """

    to_place = list(dict.fromkeys(normalized_leaf(f.basename) for f in files))
    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(normalized_leaf(f.basename), f.parse)
    batch = PlacementBatch(to_place, parsed)
    assignment = assign_episode_ids(batch, scope.target()) if scope.can_place else EpisodeAssignment(())
    assigned = assignment.assigned
    records: list[EpisodeRecord] = []
    for f in files:
        # Every placed id is in the entry's resolved set, so the index read cannot miss.
        for ep_id in assigned.get(normalized_leaf(f.basename), []):
            episode = scope.entry.by_id[ep_id]
            # An episode Sonarr sent unnumbered keys into no coverage, so it records nothing (a blanket).
            if episode.season_number is None or episode.episode_number is None:
                continue
            records.append(EpisodeRecord(season=episode.season_number, episode=episode.episode_number, size=f.size))
    return UrlPlacement(tuple(files), assignment, tuple(records), batch.all_parses_known)


class EntryPlacements(NamedTuple):
    """Every url of one entry placed, keyed as `SeadexReleaseGroupItem.urls` keys them."""

    scope: SeedScope
    """The scope every url was placed under (the seeds persist its names)."""

    by_url: Mapping[str, UrlPlacement]
    """One placement per url, every url present (an empty one for a url with no video file)."""

    @classmethod
    def place(cls, scope: SeedScope, files_by_url: Mapping[str, Sequence[SeedFile]]) -> "EntryPlacements":
        """Place each url's gathered files under one scope."""

        return cls(scope, {url: place_release(files, scope) for url, files in files_by_url.items()})

    def attach_records(self, seadex_dict: SeadexDict) -> None:
        """Write each url's placed records onto its item, and their union onto its group, for the planner."""

        for rg_item in seadex_dict.values():
            all_episodes: list[EpisodeRecord] = []
            for url, url_item in rg_item.urls.items():
                url_item.episodes = list(self.by_url[url].records)
                all_episodes.extend(url_item.episodes)
            rg_item.all_episodes = all_episodes

    def parse_failed_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a url whose placement waits on a parse Sonarr failed to serve."""

        return tuple(
            group for group, item in seadex_dict.items() if not all(self.by_url[url].parses_known for url in item.urls)
        )


@dataclass(frozen=True, slots=True)
class PendingSeedContext:
    """The per-entry values every seed built for one AniList entry carries.

    One instance per `process_al_id` call, threaded whole into
    `build_pending_seeds` (instead of loose params) and copied onto each
    `PendingImport` the entry produces.
    """

    al_id: int
    """The AniList entry id - part of each record's `PendingKey`."""
    series_id: int
    """The Sonarr series id the entry's files belong to."""
    title: str
    """The AniList display title (logging only)."""
    added_at: str
    """When the entry's seeds were written (`UPDATED_AT_STR_FORMAT`), stamped once per entry at the
    impure edge and copied onto each record for the TTL drop."""
    coverage: str | None = None
    """The entry's season/episode coverage at grab time (logging only)."""
    url: str | None = None
    """The SeaDex entry URL at grab time (logging only)."""
    guards: GuardFacts = field(default_factory=GuardFacts)
    """The plan's overwrite-guard evidence, copied onto every seed unchanged
    (see `GuardFacts`)."""


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


def build_pending_seed(
    release: SeedRelease,
    scope: SeedScope,
    entry: PendingSeedContext,
) -> PendingImport:
    """Fold one flagged release into its durable `PendingImport` seed.

    Pure: consumes the placement riding `release`, the entry's scope, and the
    per-entry context. The files were placed by the same `assign_episode_ids`
    the import wait runs, so a seed is an import-time placement made early: a
    placed file is seeded, an excluded one (another slice's, a refused
    duplicate) is recorded as never this record's to import, and anything
    held or skipped is left for import time, where the parses are re-read.
    """

    placed = release.placed
    file_episode_map = placed.assignment.assigned
    excluded_files = [placement.name for placement in placed.assignment.excluded]
    claimed = {ep_id for ids in file_episode_map.values() for ep_id in ids}

    # This record's own slice of the entry, so sibling per-episode records label
    # distinctly: the episodes its files claimed, else every episode it is verified against.
    index = scope.entry
    slice_eps = [ep for ep in index.by_id.values() if not claimed or ep.id in claimed]
    seed = PendingImport(
        infohash=release.infohash,
        series_id=entry.series_id,
        al_id=entry.al_id,
        file_episode_map=file_episode_map,
        # episode_ids is a legacy read-only fallback: never seeded (any
        # value here would only duplicate the map, which readers dedupe).
        episode_ids=[],
        release_group=release.release_group,
        is_dual_audio=release.url_item.is_dual_audio,
        seadex_files=[f.basename for f in placed.files],
        title=entry.title,
        added_at=entry.added_at,
        coverage=entry.coverage,
        url=entry.url,
        ordered_episode_ids=list(index.by_id),
        slice_coverage=coverage_string(episodes_from_ep_list(slice_eps)) or None,
        excluded_files=excluded_files,
        guards=entry.guards,
        release_sizes=list(release.url_item.size),
        names=scope.names,
    )
    # Targets that already hold a recommended file at grab time were
    # never this torrent's to insert: classify them against the record's
    # own trust slice (no sibling votes yet) so the wait's inserted
    # counts start at 0.
    grab_snapshot = EpisodeSnapshot(
        episodes=index,
        trusted=trusted_groups(seed),
        owned_episode_sizes=seed.guards.owned_sizes,
    )
    preowned = [
        ep_id
        for ep_id, status in grab_snapshot.statuses(sorted(claimed)).by_id.items()
        if status is EpisodeFileStatus.RECOMMENDED
    ]
    return replace(seed, preowned_episode_ids=preowned)


def build_pending_seeds(
    seadex_dict: SeadexDict,
    placed: EntryPlacements,
    entry: PendingSeedContext,
) -> dict[str, PendingImport]:
    """Fold every flagged url carrying an infohash and a video file into its seed, keyed by infohash. Pure."""

    seeds: dict[str, PendingImport] = {}
    for flagged in flagged_urls(seadex_dict):
        placement = placed.by_url[flagged.url]
        if not placement.files:
            continue
        release = SeedRelease(flagged.group, flagged.item, flagged.infohash, placement)
        seeds[flagged.infohash] = build_pending_seed(release, placed.scope, entry)
    return seeds
