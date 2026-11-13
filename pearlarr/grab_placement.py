"""Grab-time placement: each release's files placed for the grabbing title, the planner's episodes, and the updates."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Self

from .arr_models import ParsedFileInfo
from .coverage import coverage_string, episodes_from_ep_list
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, GrabbedGroups
from .file_paths import file_name, normalize_name
from .numberings import NumberingStep
from .pairs import unambiguous
from .pending_import import (
    NO_ORDER_ONLY,
    NO_SIZES_BY_NAME,
    FileEpisodeMap,
    OwnedAtGrab,
    PendingImport,
    TitleClaim,
    TitleNames,
    TitleWindow,
    TorrentGroup,
    joined_file_map,
    sizes_by_episode,
)
from .placement_types import (
    EMPTY_LISTING,
    NO_FILE_MAP,
    BatchPlacements,
    EpisodeBySize,
    EpisodeIndex,
    FetchedListings,
    GrabbingTitle,
    PlacementBatch,
    PlacementVerdict,
    SeadexEvidence,
    TorrentListing,
)
from .releases import (
    NO_ANOTHER_TITLES_COUNTS,
    AnotherTitlesCounts,
    EpisodeKey,
    OrderOnlyKey,
    PlannedGrab,
    Release,
    ReleaseEpisode,
    SeadexDict,
    SkipReason,
    releases_to_grab,
    season_episode_key,
)
from .window_placement import NOTHING_PLACED, WindowOrder, ordered_windows, place_across_windows, targets_of


@dataclass(frozen=True, slots=True)
class ListedFile:
    """One listed video file, its size and its cached grab-time `/parse`, as the placement reads it."""

    file_name: str
    """The raw SeaDex file name."""

    size: int
    """The listed size, carried onto the episodes the planner compares by size."""

    parse: ParsedFileInfo | None
    """Sonarr's parse of the name, or None when no fresh parse cache row exists. An unknown parse keeps the name
    number passes from placing, and skips the release while it has a file left to place (`SkipReason.INPUT_UNREAD`)."""

    name: str = field(init=False, repr=False, compare=False)
    """The normalized file name (`normalize_name` of its `file_name`): what the batch, the sizes and the stored file
    map key a file by."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_name(file_name(self.file_name)))


@dataclass(frozen=True, slots=True)
class FetchedSeadex:
    """What this run read from SeaDex about one torrent: its listing and the series' episodes by size."""

    listing: TorrentListing | None
    """None if an entry listing the torrent couldn't be read, which places nothing: the windows of the titles
    listing it are unknown."""

    episode_by_size: EpisodeBySize | None
    """None if a listing couldn't be read. The release is then skipped and the title re-checked next run."""


@dataclass(frozen=True, slots=True)
class KnownTorrent:
    """Everything the run knows about one torrent: its pending import, the series indexes, and the SeaDex evidence."""

    pending_import: PendingImport | None
    """The stored pending import this grab joins, None when the torrent is new."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A window's series left out was not read, which places nothing."""

    fetched: FetchedSeadex
    """The torrent's listing and the series' episodes by size."""

    def seadex_evidence(self, sizes: Mapping[str, int]) -> SeadexEvidence:
        """The `SeadexEvidence` every window of this torrent is placed with: this run's listing and size matches.

        `sizes` is each file's listed size by normalized name. No size matches when the episodes by size weren't read.
        """

        listing, identities = self.fetched.listing, self.fetched.episode_by_size
        identified = {} if identities is None else {n: identities[s] for n, s in sizes.items() if s in identities}
        return SeadexEvidence(EMPTY_LISTING if listing is None else listing, identified)

    def order(self, grabbing_window: TitleWindow) -> WindowOrder | None:
        """The windows the torrent is placed under, None when the listing or a window's series is unread.

        Every window of this run's listing, every stored claim's, and `grabbing_window`, the grabbing title's, in
        `ordered_windows`' order.
        """

        listing = self.fetched.listing
        if listing is None:
            return None
        return ordered_windows(
            listing, () if self.pending_import is None else self.pending_import.claims, grabbing_window, self.indexes
        )


type KnownTorrents = Mapping[str, KnownTorrent]
"""One per release of the entry, keyed by url as `GroupReleases.releases` keys them."""


def entry_hashes(seadex_dict: SeadexDict) -> frozenset[str]:
    """Each release's `qbit_hash`, the key of its stored pending import (a release with no hash has none)."""

    return frozenset(
        qbit_hash
        for group_releases in seadex_dict.values()
        for release in group_releases.releases.values()
        if (qbit_hash := release.qbit_hash) is not None
    )


@dataclass(frozen=True, slots=True)
class TorrentFetches:
    """The run's reads of the entry's torrents, each made once: the store, the series indexes, the listings."""

    pending_imports: Mapping[str, PendingImport]
    """The stored pending imports among the entry's hashes (`entry_hashes`)."""

    indexes: Mapping[int, EpisodeIndex]
    """The series indexes read this run. A claim's series left out was not read, which places nothing."""

    listings: FetchedListings | None
    """The series' SeaDex listings, None when SeaDex was unreachable."""

    def known(self, seadex_dict: SeadexDict) -> KnownTorrents:
        """What the run knows about each release's torrent, keyed by url.

        The pending import is read by the release's `qbit_hash`, the listing by SeaDex's hash.
        """

        return {
            release.url: KnownTorrent(
                None if release.qbit_hash is None else self.pending_imports.get(release.qbit_hash),
                self.indexes,
                self._fetched_seadex(release.infohash),
            )
            for group_releases in seadex_dict.values()
            for release in group_releases.releases.values()
        }

    def _fetched_seadex(self, infohash: str | None) -> FetchedSeadex:
        """The evidence for one release's torrent.

        A release with no infohash always gets `EMPTY_LISTING`, even when SeaDex is unreachable, since no listing
        could match it anyway.
        """

        if self.listings is None:
            return FetchedSeadex(EMPTY_LISTING if infohash is None else None, None)
        listing = EMPTY_LISTING if infohash is None else self.listings.listing(infohash)
        return FetchedSeadex(listing, self.listings.episode_by_size)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacedRelease:
    """One release's files placed at grab time: the verdicts, and the episodes the planner judges coverage by."""

    files: tuple[ListedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    placements: BatchPlacements
    """This placement's verdicts, one per distinct normalized file name: the whole batch when the torrent
    is new, else the files the stored pending import's file map does not cover. A file that is another listed title's
    is `ANOTHER_TITLES` (`TitleView.placements`)."""

    episodes: tuple[ReleaseEpisode, ...]
    """One `(season, episode, size)` per file and episode of the whole file map inside the title, in file order.
    A file nothing placed adds none, and a release left for a hand import (`SkipReason.left_for_hand_import`) has
    none. A release with no episodes makes the planner count its release group as covering every episode."""

    covered_ids: frozenset[int]
    """The title's episode ids the whole file map covers (a stored placement on another series adds none)."""

    inputs_read: bool
    """True when the placement had everything it needed this run: every window was built, the episodes by size were
    read, and, while a file is left to place, every parse came from Sonarr (`PlacementBatch.all_parses_known`). When
    False, a file may have waited on a parse or been placed without its size, so the title gets re-checked next run."""

    skip_reason: SkipReason | None
    """Why the release is not grabbed this run, None when it may be."""

    another_titles_counts: AnotherTitlesCounts
    """For `SkipReason.PLACED_ELSEWHERE`, how many of the files are another title's and how many nothing placed.
    `NO_ANOTHER_TITLES_COUNTS` for any other release."""

    pending_import: PendingImport | None
    """The pending import the release was placed against, None when the torrent is new."""

    intended_sizes: Mapping[int, int]
    """Each episode id of the file map to the listed size of the file placed on it.
    An id two files size differently is left out."""

    sizes_by_name: Mapping[str, int] = field(init=False, repr=False, compare=False)
    """Normalized listing name -> its listed size, read off `files` (see `_sizes_by_name`)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "intended_sizes", MappingProxyType(dict(self.intended_sizes)))
        object.__setattr__(self, "sizes_by_name", MappingProxyType(_sizes_by_name(self.files)))


def _sizes_by_name(files: Sequence[ListedFile]) -> dict[str, int]:
    """Normalized listing name -> its listed size, a name listed at two sizes left out."""

    return unambiguous((f.name, f.size) for f in files)


def listed_batch(files: Sequence[ListedFile], file_map: FileEpisodeMap = NO_FILE_MAP) -> PlacementBatch:
    """One release's files as a placement batch keyed by NORMALIZED file name, as the downloaded names are at import.

    NFC/NFD-safe: two listed files with one normalized name share its verdict (the first parse) and keep their sizes.
    `file_map` is the stored pending import's file map (`PendingImport.file_episode_map`), whose names are never
    placed.
    """

    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(f.name, f.parse)
    return PlacementBatch(list(parsed), parsed, file_map)


def place_release(files: Sequence[ListedFile], grabbing_title: GrabbingTitle, torrent: KnownTorrent) -> PlacedRelease:
    """Place one release's files under the torrent's windows (`KnownTorrent.order`), with no I/O.

    Nothing is placed when the listing or a window's series wasn't read. After the windows merge, an absolute-only
    file that they left is placed where every numbering agrees (`NumberingStep`, `WindowedPlacements.with_proposals`),
    the window holding its ids judging it. A file another title's window placed, or left for a hand import without
    involving a window the pending import imports, is set aside as that title's. The release is skipped (`SkipReason`)
    when a file of the grabbing title's window is misnumbered, when a read the placement needs failed, or when nothing
    of it lands on the title and either the title's window left a file for a hand import or some of it is another
    title's.
    """

    file_map: FileEpisodeMap = (
        NO_FILE_MAP if torrent.pending_import is None else torrent.pending_import.file_episode_map
    )
    batch = listed_batch(files, file_map=file_map)
    sizes = _sizes_by_name(files)
    order = torrent.order(grabbing_title.window)

    # 1. Place under the torrent's windows.
    view = NOTHING_PLACED
    # With no file left to place, the windows have nothing to judge.
    if batch.new_names and grabbing_title.can_place and order is not None:
        scopes = targets_of(order.windows, torrent.indexes, torrent.seadex_evidence(sizes))
        windowed = place_across_windows(batch, scopes)
        proposals = NumberingStep.of(grabbing_title, order.windows).propose(
            batch, windowed.facts[order.grabbing_window_at].names
        )
        view = windowed.with_proposals(batch, proposals).seen_by(order.grabbing_window_at, order.imports)

    # 2. What the whole file map puts on the title.
    whole_file_map = joined_file_map(file_map, view.placements.placed)
    on_window = _OnWindow.of(files, grabbing_title.title_index, whole_file_map)
    placed_on_window = bool(on_window.covered_ids)

    # 3. Whether the release is skipped.
    unread = (
        # An order that couldn't be built is a read that failed, since the release's windows are unknown.
        order is None
        or torrent.fetched.episode_by_size is None
        # Every video file's parse feeds the checks on the files being placed, those in the file map included, and
        # nothing places a file after the grab, so an unknown parse skips the release while it has a file left to
        # place. A release the pending import's file map covers whole needs no parse.
        or (bool(batch.new_names) and not batch.all_parses_known)
    )
    # With nothing of the whole file map on the title, a file another title's window placed or left for a hand import
    # is that title's, and so is a file the file map puts on an episode another title's window holds.
    another_titles = (
        frozenset[str]()
        if placed_on_window
        else view.another_titles | _on_another_titles_window(batch, whole_file_map, order)
    )
    # A release with no video file (subs, fonts) places nothing and waits on nothing.
    skip_reason = None
    if files:
        skip_reason = _skip_reason_of(
            misnumbered=view.misnumbered,
            unread=unread,
            hand_import_verdict=None if placed_on_window else view.hand_import_verdict,
            another_titles=bool(another_titles),
        )
    # With every window's series read, a grabbing title that can't place (`GrabbingTitle.can_place`) still places
    # nothing. A new torrent is then judged by group and size. A stored one's claim joins with nothing placed, a
    # grab-time verdict that would be final, so its title is re-checked.
    known = order is not None and (torrent.pending_import is None or grabbing_title.can_place)
    return PlacedRelease(
        files=tuple(files),
        placements=view.placements,
        # A release left for a hand import adds no episodes: the planner judges it by the arr's files on its window.
        episodes=() if skip_reason is not None and skip_reason.left_for_hand_import else on_window.episodes,
        covered_ids=on_window.covered_ids,
        inputs_read=not files or (known and not unread),
        skip_reason=skip_reason,
        another_titles_counts=(
            _another_titles_counts(another_titles, view.placements)
            if skip_reason is SkipReason.PLACED_ELSEWHERE
            else NO_ANOTHER_TITLES_COUNTS
        ),
        pending_import=torrent.pending_import,
        intended_sizes=sizes_by_episode(whole_file_map, sizes),
    )


def _on_another_titles_window(
    batch: PlacementBatch, whole_file_map: FileEpisodeMap, order: WindowOrder | None
) -> frozenset[str]:
    """The files to place that the whole file map puts on an episode another listed title's window holds.

    A file the file map puts on an episode no listed title's window holds is nobody's and isn't among them. Judged on
    the whole file map, so a sibling's stored files count as its fresh ones would.
    """

    if order is None:
        return frozenset()
    return frozenset(name for name in batch.to_place if not order.other_ids.isdisjoint(whole_file_map.get(name, ())))


@dataclass(frozen=True, slots=True)
class _OnWindow:
    """What the whole file map puts on the title."""

    episodes: tuple[ReleaseEpisode, ...]
    """One `(season, episode, size)` per file and episode of the file map inside the title, in file order."""
    covered_ids: frozenset[int]
    """The title's episode ids the file map covers."""

    @classmethod
    def of(cls, files: Sequence[ListedFile], index: EpisodeIndex, whole_file_map: FileEpisodeMap) -> Self:
        """What `whole_file_map` puts on `index`, read in `files` order."""

        records: list[ReleaseEpisode] = []
        claimed: set[int] = set()
        for f in files:
            for ep_id in whole_file_map.get(f.name, ()):
                # A stored id on another series is not this title's. One inside the title counts toward its coverage.
                episode = index.by_id.get(ep_id)
                if episode is None:
                    continue
                claimed.add(ep_id)
                # An episode Sonarr sent without numbers fits no coverage key, so it adds no `ReleaseEpisode`.
                if episode.season_number is None or episode.episode_number is None:
                    continue
                records.append(
                    ReleaseEpisode(season=episode.season_number, episode=episode.episode_number, size=f.size)
                )
        return cls(tuple(records), frozenset(claimed))


def _skip_reason_of(
    *, misnumbered: bool, unread: bool, hand_import_verdict: PlacementVerdict | None, another_titles: bool
) -> SkipReason | None:
    """Why the release isn't grabbed: the first of these that holds, in this order.

    Four inputs, since their order is the rule. `misnumbered`: a file of the grabbing title's window is misnumbered
    (`TitleView.misnumbered`). `unread`: a read the placement needs failed. `hand_import_verdict`: the hand-import
    verdict the grabbing title's window gave a file no window placed (`TitleView.hand_import_verdict`), when nothing
    of the whole file map lands on the title. A file episode titles contradict skips the release as `TITLE_REFUTED`,
    and a file another TVDB order reads outside every listed title as `MISNUMBERED`. It outranks `another_titles`,
    since a file the grabbing title's window left for a hand import is that title's hand import, and saying it's
    another title's would be wrong. `another_titles`: nothing of the whole file map lands on the title, and some file
    is another title's.
    """

    if misnumbered:
        return SkipReason.MISNUMBERED
    if unread:
        return SkipReason.INPUT_UNREAD
    if hand_import_verdict is PlacementVerdict.TITLE_REFUTED:
        return SkipReason.TITLE_REFUTED
    if hand_import_verdict is not None:
        return SkipReason.MISNUMBERED
    return SkipReason.PLACED_ELSEWHERE if another_titles else None


def _another_titles_counts(another_titles: frozenset[str], placements: BatchPlacements) -> AnotherTitlesCounts:
    """How a release whose files are another title's splits: `another_titles`, and the files nothing placed outside it.

    A file another title's window left for a hand import can be unplaced as well as among `another_titles` (it
    involved a claimed title), so it counts once.
    """

    unplaced = sum(name not in another_titles for name in placements.unplaced)
    return AnotherTitlesCounts(another_titles=len(another_titles), unplaced=unplaced)


@dataclass(frozen=True, slots=True)
class SizeOverName:
    """A file placed by size on an episode its name doesn't point at."""

    name: str
    """The normalized file name."""

    parsed_as: str
    """What the name reads as, for the warning: its season and episode, else Sonarr's matched episodes, else its
    absolute numbers."""

    by_size: EpisodeKey
    """The episode its size matched, where it was placed."""


def size_over_name_files(placed: PlacedRelease, series: EpisodeIndex) -> tuple[SizeOverName, ...]:
    """One per file placed by size over its name (`PlacementVerdict.SIZE_OVER_NAME`)."""

    parses = listed_batch(placed.files).parsed
    overrides: list[SizeOverName] = []
    for placement in placed.placements.placements:
        if placement.verdict is not PlacementVerdict.SIZE_OVER_NAME:
            continue
        if (episode := series.by_id.get(placement.ids[0])) is not None:
            named = season_episode_key(episode.season_number, episode.episode_number)
            overrides.append(SizeOverName(placement.name, _parsed_as(parses.get(placement.name)), named))
    return tuple(overrides)


def _parsed_as(info: ParsedFileInfo | None) -> str:
    """How the warning shows a parse: its season and episode, else Sonarr's matched episodes, else its absolutes."""

    if info is None:
        return "nothing"
    if info.season_number is not None and info.episode_numbers:
        keys = [EpisodeKey(info.season_number, episode) for episode in info.episode_numbers]
    else:
        keys = [EpisodeKey(matched.season_number, matched.episode_number) for matched in info.matched_episodes]
    labels = [key.label for key in keys] or [f"absolute {number}" for number in info.absolute_episode_numbers]
    return ", ".join(labels) or "nothing"


@dataclass(frozen=True, slots=True)
class EntryPlacements:
    """Every release of one entry placed, keyed as `GroupReleases.releases` keys them."""

    grabbing_title: GrabbingTitle
    """The grabbing title every release was placed for (the pending import updates store its names)."""

    by_url: Mapping[str, PlacedRelease]
    """One `PlacedRelease` per url, every release present (an empty one for a release with no video file)."""

    @classmethod
    def place(
        cls,
        grabbing_title: GrabbingTitle,
        files_by_url: Mapping[str, Sequence[ListedFile]],
        torrents: KnownTorrents,
    ) -> Self:
        """Place each release's files for the grabbing title, against what the run knows of its torrent."""

        return cls(
            grabbing_title,
            {url: place_release(files, grabbing_title, torrents[url]) for url, files in files_by_url.items()},
        )

    def attach_placements(self, seadex_dict: SeadexDict) -> None:
        """Write each release's episodes, skip reason, and skip counts onto it, and their episodes onto its group."""

        for group_releases in seadex_dict.values():
            all_episodes: list[ReleaseEpisode] = []
            for url, release in group_releases.releases.items():
                placement = self.by_url[url]
                release.episodes = list(placement.episodes)
                release.skip_reason = placement.skip_reason
                release.another_titles_counts = placement.another_titles_counts
                all_episodes.extend(release.episodes)
            group_releases.all_episodes = all_episodes

    def input_unread_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a release whose placement waits on a read that failed."""

        return tuple(
            group
            for group, item in seadex_dict.items()
            if not all(self.by_url[url].inputs_read for url in item.releases)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimTerms:
    """What one claim records about a torrent: its window's ids and names, its preowned ids, and its claimed coverage.

    All empty with no window.
    """

    ordered_episode_ids: tuple[int, ...] = ()
    title_names: TitleNames = field(default_factory=TitleNames)
    preowned_episode_ids: tuple[int, ...] = ()
    claimed_coverage: str | None = None


NO_WINDOW_TERMS = ClaimTerms()
"""The terms of a claim whose import reads nothing but the infohash (a Radarr grab)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TitleFacts:
    """What a title is, independent of any torrent: the fields every claim it makes carries."""

    al_id: int
    """The title's AniList id, the claim's key."""
    series_id: int
    """The Sonarr series id the title's files belong to (0 for a Radarr movie)."""
    title: str
    """The AniList display name (logging only)."""
    coverage: str | None
    """The title's season/episode coverage at grab time (logging only)."""
    url: str | None
    """The SeaDex entry URL at grab time (logging only)."""
    guards: OwnedAtGrab
    """The plan's overwrite-guard evidence, copied onto every claim unchanged (see `OwnedAtGrab`)."""

    def claim(self, terms: ClaimTerms) -> TitleClaim:
        """The title's claim on `terms`."""

        return TitleClaim(
            al_id=self.al_id,
            series_id=self.series_id,
            title=self.title,
            window_coverage=self.coverage,
            url=self.url,
            ordered_episode_ids=terms.ordered_episode_ids,
            names=terms.title_names,
            preowned_episode_ids=terms.preowned_episode_ids,
            claimed_coverage=terms.claimed_coverage,
            guards=self.guards,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TorrentFacts:
    """What a release's torrent is, independent of any title: the pending import's identity fields."""

    infohash: str
    """The qBittorrent tracking key, lowercase."""
    release_group: str
    """The SeaDex release group (authoritative)."""
    is_dual_audio: bool
    """The listing's dual-audio flag."""
    seadex_files: tuple[str, ...]
    """The listing's video file names, the import's progress denominator."""
    release_sizes: tuple[int, ...]
    """The listing's file sizes, the sizes the torrent's group is trusted at (`trusted_groups`)."""
    sizes_by_name: Mapping[str, int]
    """Normalized listing name -> its listed size, the file each placed episode should have (a name listed at two
    sizes is left out)."""
    is_cross_seed: bool = False
    """Whether the torrent is a cross-seed of the listing's (`PendingImport.is_cross_seed`)."""


@dataclass(frozen=True, slots=True)
class PlacedGrab:
    """One planned grab's update inputs: the release and its placed files (I/O done)."""

    release_group: str
    """The SeaDex release group (authoritative)."""

    release: Release
    """The release to grab, whole: the update reads its sizes and dual-audio flag."""

    qbit_hash: str
    """The qBittorrent tracking key (the release's `qbit_hash`, already narrowed to `str`)."""

    placed: PlacedRelease
    """The release's files placed at grab time (`place_release`)."""

    @property
    def torrent_group(self) -> TorrentGroup:
        """The release's group at its listing's sizes, the trusted group `trusted_groups` sets last."""

        return TorrentGroup(self.release_group, tuple(self.release.file_sizes))

    @property
    def facts(self) -> TorrentFacts:
        """The torrent's identity fields as the pending import stores them."""

        return TorrentFacts(
            infohash=self.qbit_hash,
            release_group=self.release_group,
            is_dual_audio=self.release.is_dual_audio,
            seadex_files=tuple(f.file_name for f in self.placed.files),
            release_sizes=self.torrent_group.sizes,
            sizes_by_name=self.placed.sizes_by_name,
            is_cross_seed=self.release.cross_seed_hash is not None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingImportUpdate:
    """One title's contribution to a torrent's pending import."""

    facts: TorrentFacts
    """The torrent's identity fields for a new pending import. A stored one keeps its own."""

    file_map: FileEpisodeMap
    """This placement's fresh file map: the whole batch when the torrent is new, else the files the stored file map
    lacks."""

    excluded: tuple[str, ...]
    """The names this placement proved aren't the pending import's to import under its current claims."""

    hand_import_files: tuple[str, ...]
    """The names this placement left for a hand import (`BatchPlacements.hand_import_files`), which keep the
    pending import stored once the placed files are in."""

    order_only: Mapping[str, OrderOnlyKey]
    """The keys of those names that only another TVDB order numbers (`BatchPlacements.order_only`)."""

    claim: TitleClaim
    """This title's fresh claim."""

    pending_import: PendingImport | None
    """The stored pending import this update joins, None when the torrent is new."""

    def joined(self) -> PendingImport | None:
        """The stored pending import with this grab joined, None when the torrent is new."""

        if self.pending_import is None:
            return None
        # The hand-import files join the stored ones rather than replace them: a grab whose title can't place
        # carries none, and would otherwise clear an earlier grab's.
        return (
            self.pending_import.with_placements(self.file_map)
            .with_exclusions(self.excluded)
            .with_hand_import_files(self.hand_import_files, self.order_only)
            .with_claim(self.claim)
            .with_sizes_by_name(self.facts.sizes_by_name)
        )

    def pending_import_at(self, stamp: str, *, fresh: bool) -> PendingImport:
        """The pending import to store: a new one starting at `stamp`, or the stored one with this grab joined.

        `fresh` restarts the stored one at `stamp`: a torrent added afresh downloads anew.
        """

        joined = self.joined()
        if joined is None:
            return PendingImport(
                infohash=self.facts.infohash,
                release_group=self.facts.release_group,
                is_dual_audio=self.facts.is_dual_audio,
                seadex_files=self.facts.seadex_files,
                added_at=stamp,
                file_episode_map=self.file_map,
                claims=(self.claim,),
                excluded_files=self.excluded,
                release_sizes=self.facts.release_sizes,
                sizes_by_name=self.facts.sizes_by_name,
                hand_import_files=self.hand_import_files,
                order_only=self.order_only,
                is_cross_seed=self.facts.is_cross_seed,
            )
        return replace(joined, added_at=stamp) if fresh else joined


def build_title_claim(release: PlacedGrab, grabbing_title: GrabbingTitle, facts: TitleFacts) -> TitleClaim:
    """The title's claim on the release: its window, claimed coverage, and preowned ids. Pure."""

    claimed = release.placed.covered_ids
    index = grabbing_title.title_index
    # This claim's part of the title, so sibling titles' claims label distinctly: the episodes its files cover,
    # else every episode it is checked against.
    claimed_episodes = [ep for ep in index.by_id.values() if not claimed or ep.id in claimed]
    # Episodes that already have a trusted file at grab time were never this torrent's to import: classify them
    # against the claim's trusted groups (no siblings' groups yet) so the wait pass's counts of trusted files start
    # at 0. A replaced claim keeps its first preowned ids (`PendingImport.with_claim`).
    grab_snapshot = EpisodeSnapshot.guarded(index, facts.guards, GrabbedGroups(release.torrent_group))
    preowned = tuple(
        ep_id
        for ep_id, status in grab_snapshot.statuses(sorted(claimed), release.placed.intended_sizes).by_id.items()
        if status is EpisodeFileStatus.TRUSTED
    )
    claimed_coverage = coverage_string(episodes_from_ep_list(claimed_episodes)) or None
    window = grabbing_title.window
    return facts.claim(
        ClaimTerms(
            ordered_episode_ids=window.ordered_episode_ids,
            title_names=window.names,
            preowned_episode_ids=preowned,
            claimed_coverage=claimed_coverage,
        )
    )


def build_pending_import_update(
    release: PlacedGrab, grabbing_title: GrabbingTitle, facts: TitleFacts
) -> PendingImportUpdate:
    """Turn one placed release into the title's pending import update on its torrent. Pure, never places.

    A placed file goes in the file map, an excluded one is recorded as not the pending import's under its current
    claims, and one left for a hand import is recorded so the import can report it. An unplaced file is recorded
    nowhere: the import warns of it as a file in the download folder the file map lacks.
    """

    placed = release.placed
    return PendingImportUpdate(
        facts=release.facts,
        file_map=placed.placements.placed,
        excluded=tuple(placement.name for placement in placed.placements.excluded),
        hand_import_files=placed.placements.hand_import_files,
        order_only=placed.placements.order_only,
        claim=build_title_claim(release, grabbing_title, facts),
        pending_import=placed.pending_import,
    )


def build_update_without_window(
    planned: PlannedGrab, facts: TitleFacts, stored: PendingImport | None
) -> PendingImportUpdate:
    """The update for a torrent whose import reads only its hash in qBittorrent (a Radarr grab): an id-less claim."""

    return PendingImportUpdate(
        facts=TorrentFacts(
            infohash=planned.qbit_hash,
            release_group=planned.group,
            is_dual_audio=planned.item.is_dual_audio,
            seadex_files=(),
            release_sizes=(),
            sizes_by_name=NO_SIZES_BY_NAME,
            is_cross_seed=planned.item.cross_seed_hash is not None,
        ),
        file_map={},
        excluded=(),
        hand_import_files=(),
        order_only=NO_ORDER_ONLY,
        claim=facts.claim(NO_WINDOW_TERMS),
        pending_import=stored,
    )


def build_pending_import_updates(
    seadex_dict: SeadexDict,
    placed: EntryPlacements,
    facts: TitleFacts,
) -> dict[str, PendingImportUpdate]:
    """Turn every release to grab with an infohash and a video file into its update, keyed by its hash in qBittorrent.

    Pure.
    """

    updates: dict[str, PendingImportUpdate] = {}
    for planned in releases_to_grab(seadex_dict):
        placement = placed.by_url[planned.url]
        if not placement.files:
            continue
        release = PlacedGrab(planned.group, planned.item, planned.qbit_hash, placement)
        updates[planned.qbit_hash] = build_pending_import_update(release, placed.grabbing_title, facts)
    return updates
