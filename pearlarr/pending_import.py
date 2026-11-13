"""The pending import record each grab leaves for the wait pass, with its claims, windows and seed coverage."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, NamedTuple

from .arr_models import SonarrEpisode
from .file_paths import normalized_leaf
from .pairs import unambiguous
from .releases import OrderOnlyKey

type FileEpisodeMap = Mapping[str, tuple[int, ...]]
"""A read view of normalized basename -> episode ids: a record's map, a placement's, or both joined (`joined_map`).
Producers normalize the keys, and readers take them as given."""

NO_ORDER_ONLY: Mapping[str, OrderOnlyKey] = MappingProxyType({})
"""No file keyed by another TVDB order."""


def _normalized_names(names: Iterable[str]) -> set[str]:
    """Normalized-leaf SET, deliberately not a multiset."""

    return {normalized_leaf(name) for name in names}


def _normalized_names_in_order(names: Iterable[str]) -> tuple[str, ...]:
    """Each name's normalized leaf, once, in first-seen order."""

    return tuple(dict.fromkeys(normalized_leaf(name) for name in names))


NO_SIZES_BY_NAME: Mapping[str, int] = MappingProxyType({})
"""No listed sizes: a record written before they were kept, or a torrent whose files are never listed."""


def sizes_by_episode(mapped: FileEpisodeMap, listed: Mapping[str, int]) -> dict[int, int]:
    """Each mapped episode id to its file's listed size.

    An unlisted name sizes nothing, and an ambiguous id is dropped.
    """

    return unambiguous((ep_id, listed[name]) for name, ids in mapped.items() if name in listed for ep_id in ids)


def _normalized_map(entries: FileEpisodeMap) -> dict[str, tuple[int, ...]]:
    """The map with every key normalized and zero ids dropped, minus the entries that leaves empty.

    Of two keys that normalize alike, the first one left with ids wins.
    """

    normalized: dict[str, tuple[int, ...]] = {}
    for name, ids in entries.items():
        clean = tuple(i for i in ids if i)
        if clean:
            normalized.setdefault(normalized_leaf(name), clean)
    return normalized


def _normalized_keys[V](by_name: Mapping[str, V]) -> dict[str, V]:
    """`by_name` keyed by normalized leaf.

    A name whose spellings disagree on the value is dropped, unlike the map's first-wins fold: a file with no size
    is never read as misplaced and one with no key prints its bare name, where a guessed value could misjudge it.
    """

    return unambiguous((normalized_leaf(name), value) for name, value in by_name.items())


def joined_map(base: FileEpisodeMap, added: FileEpisodeMap) -> dict[str, tuple[int, ...]]:
    """`base`, then the entries of `added` whose names it lacks."""

    return {**base, **{name: ids for name, ids in added.items() if name not in base}}


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedCoverage:
    """How far a record's map and exclusions account for its listed files. Both False when it lists none."""

    mapped: bool
    """Whether the map alone names every listed file."""

    accounted: bool
    """Whether the map and the exclusions together name every listed file."""


class SizeMatchedEpisode(NamedTuple):
    """One grab-time ownership claim over an untagged on-disk file.

    A tuple on purpose: `GuardFacts.owned_sizes` and the JSON form read it as an (id, size) pair.
    """

    ep_id: int
    size: int


@dataclass(frozen=True, slots=True)
class TorrentGroup:
    """A torrent's own release group and the sizes its current listing carries (the trust policy's last vote)."""

    release_group: str
    sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TitleNames:
    """What names an entry: the arr series title and the AniList titles (English first, then romaji).

    The placement's tie-break between look-alike files scores on the words
    an AniList title adds to the series title, so both ride together.
    """

    series: str = ""
    anilist: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """The plain dict persisted inside a claim."""

        return {"series": self.series, "anilist": list(self.anilist)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "TitleNames":
        """Rebuild from the persisted dict (a record written before the names were kept reads empty)."""

        return cls(series=raw.get("series", ""), anilist=tuple(raw.get("anilist", [])))


@dataclass(frozen=True, slots=True, kw_only=True)
class TitleWindow:
    """One entry's episode window on a series: its episode ids and the names it goes by."""

    al_id: int
    """The AniList entry id."""

    series_id: int
    """The Sonarr series id the window's episodes belong to."""

    ordered_episode_ids: tuple[int, ...]
    """The window's episode ids in season order, each once. Empty when the entry's episode list came back empty."""

    names: TitleNames
    """The entry's series and AniList titles (see `EntryNames`)."""

    ignored: bool = False
    """Whether the entry was in `ignore_anilist_ids` at the read. `ordered_windows` runs an ignored window after the
    kept windows of the same kind (season or specials), so a file a kept title can place is the kept title's. A
    claim's window takes the flag from its entry's listed window."""

    episode_count: int | None = None
    """AniList's episode count for the entry at the read, None when AniList gives none. Never persisted: a claim's
    window takes it from its entry's listed window, and has none when the entry isn't listed."""

    @classmethod
    def of(
        cls,
        *,
        al_id: int,
        series_id: int,
        ep_list: Iterable[SonarrEpisode],
        names: TitleNames,
        ignored: bool = False,
        episode_count: int | None = None,
    ) -> "TitleWindow":
        """The window over `ep_list`: each episode with an id, once, in the list's order."""

        ids = tuple(dict.fromkeys(ep.id for ep in ep_list if ep.id))
        return cls(
            al_id=al_id,
            series_id=series_id,
            ordered_episode_ids=ids,
            names=names,
            ignored=ignored,
            episode_count=episode_count,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OwnedAtGrab:
    """The plan's per-entry overwrite-guard evidence, carried whole."""

    entry_groups: tuple[str, ...] = ()
    """Pick groups the plan verified current by size."""

    stale_groups: tuple[str, ...] = ()
    """Pick groups the plan judged stale on disk."""

    owned_episodes: tuple[SizeMatchedEpisode, ...] = ()
    """Episodes where the on-disk file has no release group, but matched a pick's listed size exactly."""

    @property
    def owned_sizes(self) -> dict[int, int]:
        """`owned_episodes` as the id -> size mapping the classifier reads."""

        return dict(self.owned_episodes)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "OwnedAtGrab":
        """Rebuild from the persisted dict (missing keys fall back empty)."""

        return cls(
            entry_groups=tuple(raw.get("entry_groups", [])),
            stale_groups=tuple(raw.get("stale_groups", [])),
            owned_episodes=tuple(SizeMatchedEpisode(pair[0], pair[1]) for pair in raw.get("owned_episodes", [])),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TitleClaim:
    """One AniList entry's claim on a torrent: the window its files are judged under and the slice it intends."""

    al_id: int
    """The AniList entry id, also the key of the entry's `guard_facts` row."""

    series_id: int
    """The Sonarr series id the entry's files belong to (0 on Radarr, which has no series windows)."""

    title: str | None
    """Display title (logging only)."""

    window_coverage: str | None
    """The entry's season/episode coverage when it claimed (e.g. `"S01 E01-E13"`, logging only)."""

    url: str | None
    """The SeaDex entry URL when it claimed, for the carried-over record's inline `link` line."""

    ordered_episode_ids: tuple[int, ...]
    """The entry's resolved episode ids in season order. Empty means unscoped: any id the series map holds."""

    names: TitleNames
    """The series and AniList titles the placement breaks ties by (see `EntryNames`)."""

    preowned_episode_ids: tuple[int, ...]
    """Claimed ids that already held a recommended file at the FIRST claim (the wait bar's net-out)."""

    claimed_coverage: str | None
    """This claim's own ids as a coverage string (e.g. `"S02 E06"`)."""

    guards: OwnedAtGrab = field(default_factory=OwnedAtGrab)
    """The plan's overwrite-guard evidence. Not serialized: hydrated from `guard_facts` by `al_id`."""

    def admits(self, ep_id: int) -> bool:
        """Whether the claim's window holds `ep_id`.

        An unscoped claim admits every id, though `RecordSnapshot.route` judges an id under an
        unscoped claim only when it is alone on its series.
        """

        return not self.ordered_episode_ids or ep_id in self.ordered_episode_ids

    @property
    def window(self) -> TitleWindow:
        """The claim as a window, the shape a listing holds.

        A claim with no ids gives a window with no ids, and `admits` still reads that claim as covering any id. The
        window is never flagged ignored and has no episode count: a claim predates any ignore and doesn't keep the
        count, so `ordered_windows` takes both from the listing.
        """

        return TitleWindow(
            al_id=self.al_id, series_id=self.series_id, ordered_episode_ids=self.ordered_episode_ids, names=self.names
        )

    def to_json(self) -> dict[str, Any]:
        """The plain dict persisted inside the record (the guards ride their own row): the schema, spelled out."""

        return {
            "al_id": self.al_id,
            "series_id": self.series_id,
            "title": self.title,
            "coverage": self.window_coverage,
            "url": self.url,
            "ordered_episode_ids": list(self.ordered_episode_ids),
            "names": self.names.to_json(),
            "preowned_episode_ids": list(self.preowned_episode_ids),
            "slice_coverage": self.claimed_coverage,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, guards: Mapping[int, OwnedAtGrab]) -> "TitleClaim":
        """Rebuild a claim from its persisted dict, fed the arr's guard rows by entry."""

        al_id = raw.get("al_id", 0)
        return cls(
            al_id=al_id,
            series_id=raw.get("series_id", 0),
            title=raw.get("title"),
            window_coverage=raw.get("coverage"),
            url=raw.get("url"),
            ordered_episode_ids=tuple(raw.get("ordered_episode_ids", [])),
            names=TitleNames.from_json(raw.get("names", {})),
            preowned_episode_ids=tuple(raw.get("preowned_episode_ids", [])),
            claimed_coverage=raw.get("slice_coverage"),
            guards=guards.get(al_id, OwnedAtGrab()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingImport:
    """One torrent awaiting import: its whole file map and every entry's claim on it.

    A `with_*` method returns this record itself when the change leaves it equal, so a caller tells a no-op by
    identity.
    """

    infohash: str
    """The qBittorrent tracking key, lowercase (never None). Also the dedup `downloadId` sent to Sonarr."""

    release_group: str
    """The SeaDex release group (authoritative)."""

    is_dual_audio: bool
    """Whether the SeaDex release is dual-audio. Selects the dual vs. single language list."""

    seadex_files: tuple[str, ...]
    """SeaDex filenames, for our regex quality parse."""

    added_at: str
    """When the download started, in `UPDATED_AT_STR_FORMAT`: Pearlarr's own add, or for a torrent already in
    qBittorrent the client's add time (now when it reports none). The rescan proof and Radarr's history window
    count from it."""

    file_episode_map: FileEpisodeMap
    """Normalized basename -> Sonarr episode ids over the WHOLE torrent: the grab-time seeds plus the files Sonarr's
    history placed for a download no scan saw. Folded as the record is built (`_normalized_map`) and wrapped
    read-only."""

    claims: tuple[TitleClaim, ...]
    """Every entry's claim on the torrent, in accretion order."""

    excluded_files: tuple[str, ...] = ()
    """Normalized basenames of video files that aren't this record's to import under its current claims (a
    collision-refused duplicate, a file every window resolves outside, one another listed title's window placed or
    left for a hand import), found at grab time. They only quiet the import's warning: a later claim's grab can
    still place one. Folded and deduplicated in order as the record is built, and never a name the map holds."""

    release_sizes: tuple[int, ...] = ()
    """The grabbed listing's file sizes. Lets the import tell this release's own files from a stale
    same-group copy."""

    awaiting_cleanup: bool = False
    """The import verified but a post-import effect (category move / queue close) still needs to run."""

    sizes_by_name: Mapping[str, int] = NO_SIZES_BY_NAME
    """Normalized listing name -> its listed size, so the import knows which file each episode should hold.
    Folded as the record is built (`_normalized_keys`). Empty on a row written before it was kept, which then
    never reads a file as misplaced."""

    hand_import_files: tuple[str, ...] = ()
    """Normalized basenames of the video files the grab left for a hand import (misnumbered, refuted by their
    run's titles, or numbered by another TVDB order onto no episode it could place), which keep the record pending
    once the placed files are in. Folded and deduplicated in order as
    the record is built, never a name the map holds, and a name here is never excluded. Empty on rows written before
    it was kept."""

    order_only: Mapping[str, OrderOnlyKey] = NO_ORDER_ONLY
    """Normalized basename -> its key, for each hand-import file that only another TVDB order numbers. Folded as the
    record is built to names among `hand_import_files`, and wrapped read-only. Empty on rows written before it was
    kept."""

    is_cross_seed: bool = False
    """Whether the torrent is a cross-seed: it held the release's files under another hash than SeaDex's
    (`SeadexUrlItem.cross_seed_hash`), so Pearlarr never added it. A forced `move` import of it runs as `auto`."""

    def __post_init__(self) -> None:
        # Fold the map, the exclusions, the hand-import files, their order-only keys and the by-name sizes so readers
        # can take the names as normalized, the map's ids as non-zero, and no name as two of mapped, excluded and left
        # for a hand import. The map wins, then the hand import: a title's own window can leave a file for a hand
        # import that a sibling's earlier grab excluded as another title's, and the record must still say so. Detach
        # the maps and wrap them read-only.
        mapped = _normalized_map(self.file_episode_map)
        hand_import = tuple(name for name in _normalized_names_in_order(self.hand_import_files) if name not in mapped)
        excluded = (
            name
            for name in _normalized_names_in_order(self.excluded_files)
            if name not in mapped and name not in hand_import
        )
        order_only = {name: key for name, key in _normalized_keys(self.order_only).items() if name in hand_import}
        object.__setattr__(self, "file_episode_map", MappingProxyType(mapped))
        object.__setattr__(self, "excluded_files", tuple(excluded))
        object.__setattr__(self, "hand_import_files", hand_import)
        object.__setattr__(self, "order_only", MappingProxyType(order_only))
        object.__setattr__(self, "sizes_by_name", MappingProxyType(_normalized_keys(self.sizes_by_name)))

    @property
    def own_group(self) -> TorrentGroup:
        """The torrent's own group at its listing's sizes, the trust policy's last vote."""

        return TorrentGroup(self.release_group, self.release_sizes)

    @property
    def series_ids(self) -> tuple[int, ...]:
        """The distinct series the claims span, claim order."""

        return tuple(dict.fromkeys(claim.series_id for claim in self.claims))

    def claim_for(self, series_id: int) -> TitleClaim:
        """The first claim on `series_id` (the caller reads a record stored under that series)."""

        claim = next((claim for claim in self.claims if claim.series_id == series_id), None)
        if claim is None:
            raise LookupError(f"record {self.infohash} has no claim on series {series_id}")
        return claim

    @property
    def display_label(self) -> str:
        """The cockpit/ledger/report row label: `titles · group[ · episode slices]`, every claim named."""

        base = " & ".join(dict.fromkeys(claim.title for claim in self.claims if claim.title)) or self.infohash
        if self.release_group:
            base = f"{base} · {self.release_group}"
        if slices := ", ".join(dict.fromkeys(c.claimed_coverage for c in self.claims if c.claimed_coverage)):
            base = f"{base} · {slices}"
        return base

    def file_map_episode_ids(self) -> list[int]:
        """Our intended episode ids: the map's values in first-claim order."""

        return list(dict.fromkeys(ep_id for file_ids in self.file_episode_map.values() for ep_id in file_ids))

    def window_episode_ids(self) -> list[int]:
        """The episodes a hand import must fill (`filled_by_hand`): the claims' windows in order, else the seeds' ids."""

        windows = list(dict.fromkeys(ep_id for claim in self.claims for ep_id in claim.ordered_episode_ids))
        return windows or sorted(self.file_map_episode_ids())

    def guards_for(self, series_id: int) -> OwnedAtGrab:
        """The guard evidence of every claim on `series_id` as one.

        The groups are unioned in claim order and the owned episodes concatenated.
        """

        entry_groups: dict[str, None] = {}
        stale_groups: dict[str, None] = {}
        owned: list[SizeMatchedEpisode] = []
        for claim in self.claims:
            if claim.series_id == series_id:
                entry_groups.update(dict.fromkeys(claim.guards.entry_groups))
                stale_groups.update(dict.fromkeys(claim.guards.stale_groups))
                owned.extend(claim.guards.owned_episodes)
        return OwnedAtGrab(
            entry_groups=tuple(entry_groups), stale_groups=tuple(stale_groups), owned_episodes=tuple(owned)
        )

    def seed_coverage(self) -> SeedCoverage:
        """Coverage from normalized-name SUPERSETS (never lengths): a healed extra can't fake it."""

        needed = _normalized_names(self.seadex_files)
        if not needed:
            return SeedCoverage(mapped=False, accounted=False)
        mapped_names = self.file_episode_map.keys()
        if mapped_names >= needed:
            return SeedCoverage(mapped=True, accounted=True)
        covered = mapped_names | set(self.excluded_files)
        return SeedCoverage(mapped=False, accounted=covered >= needed)

    def unplaced_names(self) -> set[str]:
        """Normalized listing names the map does not cover and no exclusion claims: what a rebuild may still place."""

        return _normalized_names(self.seadex_files) - self.file_episode_map.keys() - set(self.excluded_files)

    def _or_self(self, changed: "PendingImport") -> "PendingImport":
        """`changed`, or this record when the change left it equal."""

        return self if changed == self else changed

    def with_placements(self, placements: FileEpisodeMap) -> "PendingImport":
        """The record with `placements` joined onto its map (`joined_map`).

        Their names are lifted out of the exclusions and the hand-import files.
        """

        return self._or_self(replace(self, file_episode_map=joined_map(self.file_episode_map, placements)))

    def with_exclusions(self, names: Iterable[str]) -> "PendingImport":
        """The record with `names` appended to its exclusions, less any the map or the hand-import files hold."""

        return self._or_self(replace(self, excluded_files=(*self.excluded_files, *names)))

    def with_sizes_by_name(self, sizes: Mapping[str, int]) -> "PendingImport":
        """The record with `sizes` learned when it has none, else itself.

        Only an empty map fills: an infohash's listing never changes, and an unscoped seed carries none.
        """

        return self._or_self(replace(self, sizes_by_name=self.sizes_by_name or sizes))

    def with_hand_import_files(self, names: Iterable[str], order_only: Mapping[str, OrderOnlyKey]) -> "PendingImport":
        """The record with `names` joined to its hand-import files and lifted out of its exclusions.

        `order_only` holds their keys. A name the record already keys keeps its own.
        """

        joined = replace(
            self,
            hand_import_files=(*self.hand_import_files, *names),
            order_only={**_normalized_keys(order_only), **self.order_only},
        )
        return self._or_self(joined)

    def with_claim(self, claim: TitleClaim) -> "PendingImport":
        """The record with `claim` joined and its cleanup flag cleared (new files to import make it active again).

        A re-flag by an entry already claiming replaces its claim, keeping the stored claim's preowned ids: the
        window follows the newest plan, the wait bar's net-out never restarts.
        """

        claims = (*self.claims, claim)
        for index, stored in enumerate(self.claims):
            if stored.al_id == claim.al_id:
                kept = replace(claim, preowned_episode_ids=stored.preowned_episode_ids)
                claims = (*self.claims[:index], kept, *self.claims[index + 1 :])
                break
        return self._or_self(replace(self, claims=claims, awaiting_cleanup=False))

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under `pending_imports`."""

        return {
            "infohash": self.infohash,
            "release_group": self.release_group,
            "is_dual_audio": self.is_dual_audio,
            "seadex_files": list(self.seadex_files),
            "added_at": self.added_at,
            "file_episode_map": {name: list(ids) for name, ids in self.file_episode_map.items()},
            "claims": [claim.to_json() for claim in self.claims],
            "excluded_files": list(self.excluded_files),
            "release_sizes": list(self.release_sizes),
            "awaiting_cleanup": self.awaiting_cleanup,
            "sizes_by_name": dict(self.sizes_by_name),
            "hand_import_files": list(self.hand_import_files),
            "order_only": {name: key.to_json() for name, key in self.order_only.items()},
            "cross_seed": self.is_cross_seed,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, guards: Mapping[int, OwnedAtGrab]) -> "PendingImport":
        """Rebuild a record from its persisted cache-store dict, each claim fed its entry's guard row."""

        seadex_files = tuple(raw.get("seadex_files", []))
        mapped = {name: tuple(ids) for name, ids in raw.get("file_episode_map", {}).items()}
        excluded = tuple(raw.get("excluded_files", []))
        return cls(
            infohash=raw.get("infohash", ""),
            release_group=raw.get("release_group", ""),
            is_dual_audio=raw.get("is_dual_audio", False),
            seadex_files=seadex_files,
            added_at=added_at_of(raw),
            file_episode_map=mapped,
            claims=tuple(TitleClaim.from_json(claim, guards=guards) for claim in raw.get("claims", [])),
            excluded_files=excluded,
            release_sizes=tuple(raw.get("release_sizes", [])),
            awaiting_cleanup=is_awaiting_cleanup(raw),
            sizes_by_name=raw.get("sizes_by_name", {}),
            hand_import_files=(
                tuple(raw["hand_import_files"])
                if "hand_import_files" in raw
                else _unplaced_listed_files(seadex_files, mapped, excluded)
            ),
            order_only={
                name: key
                for name, entry in raw.get("order_only", {}).items()
                if (key := OrderOnlyKey.from_json(entry)) is not None
            },
            is_cross_seed=raw.get("cross_seed", False),
        )


def _unplaced_listed_files(
    listed: Iterable[str], mapped: Mapping[str, tuple[int, ...]], excluded: Iterable[str]
) -> tuple[str, ...]:
    """The listed files a stored row with no `hand_import_files` key left for a hand import.

    Every one the grab neither mapped nor excluded counts, since nothing else on the row tells a refused file from a
    plain skip.
    """

    taken = {*_normalized_names_in_order(mapped), *_normalized_names_in_order(excluded)}
    return tuple(name for name in _normalized_names_in_order(listed) if name not in taken)


def is_awaiting_cleanup(raw: dict[str, Any]) -> bool:
    """The cleanup flag off a stored row's raw dict, sparing keys-only readers a rehydration."""

    return bool(raw.get("awaiting_cleanup"))


def added_at_of(raw: dict[str, Any]) -> str:
    """A stored row's `added_at` off its raw dict (`""` when unset or not a string)."""

    stamp = raw.get("added_at", "")
    return stamp if isinstance(stamp, str) else ""


def hydrate_pending(rows: Mapping[str, dict[str, Any]], guards: Mapping[int, OwnedAtGrab]) -> dict[str, PendingImport]:
    """Rehydrate stored rows into records, keyed by infohash, each claim fed its entry's guard row."""

    return {key: PendingImport.from_json(raw, guards=guards) for key, raw in rows.items()}
