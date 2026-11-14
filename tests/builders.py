# pyright: strict
# pyright: reportPrivateUsage=false
# The factories assemble objects by their private collaborator fields (strat._episodes), which strict re-flags.
"""Builders and a bare-instance factory for the characterization tests."""

import dataclasses
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, override

import httpx
import qbittorrentapi
from pydantic import BaseModel
from seadex import EntryRecord, File, Tag, TorrentRecord, Tracker

from pearlarr.anibridge import NO_ALIASES, SpecialAliases
from pearlarr.anilist_client import MAX_RETRIES, AniListAnswers, AniListBody, AniListClient
from pearlarr.anilist_gateway import AniListGateway
from pearlarr.arr_categories import _CATEGORY_FIELDS, ArrCategoryResolver, CategorySources, _read_clients
from pearlarr.arr_http import ArrHttp
from pearlarr.arr_models import (
    ArrFiles,
    DownloadClientRecord,
    ManualImportCandidate,
    MatchedEpisode,
    ParsedFileInfo,
    QueueRecord,
    SonarrEpisode,
)
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.cache import (
    _CACHED_TITLE_COLUMNS,
    AbstractCacheStore,
    CachedTitle,
    CachedTitleUpdate,
    CacheStats,
    HistoryCheckpoint,
    selection_hash_key,
)
from pearlarr.clock import Clock
from pearlarr.config import AppConfig, Arr
from pearlarr.episode_state import EpisodeFileStatus, EpisodeSnapshot, EpisodeStatuses, TrustedGroups
from pearlarr.grab_pipeline import GrabPipeline, GrabRequest
from pearlarr.grab_placement import FetchedSeadex, KnownTorrent, PendingImportUpdate, TitleFacts, TorrentFacts
from pearlarr.import_wait import ImportWaitManager, PostImportCleanup, WaitPollers
from pearlarr.mappings import MappingResolver, MappingSources
from pearlarr.notify import Notifier
from pearlarr.output import SeverityCounter, emit_to_hub
from pearlarr.parse_cache import ParseCacheRows
from pearlarr.pending_import import (
    NO_ORDER_ONLY,
    NO_SIZES_BY_NAME,
    FileEpisodeMap,
    OwnedAtGrab,
    PendingImport,
    TitleClaim,
    TitleNames,
    TitleWindow,
    TorrentGroup,
)
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.placement_types import (
    EMPTY_LISTING,
    NO_FILE_MAP,
    BatchPlacements,
    EpisodeBySize,
    EpisodeIndex,
    GrabbingTitle,
    PlacementBatch,
    PlacementTarget,
    PlacementVerdict,
    TorrentListing,
    episode_index,
)
from pearlarr.placer import place_batch
from pearlarr.planner import GrabPlan, GrabPlanner, PrivateOnlyNotices
from pearlarr.progress import ProgressCallback
from pearlarr.radarr_client import AbstractRadarrClient
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.releases import EpisodeKey, GroupReleases, OrderOnlyKey, Release, ReleaseEpisode, SeadexDict
from pearlarr.reporter import RunReporter
from pearlarr.seadex_filter import SeadexReleaseFilter
from pearlarr.seadex_gateway import SeaDexNoEntry, SeaDexSource
from pearlarr.skyhook_client import AbstractSkyHookClient
from pearlarr.skyhook_gateway import SkyHookGateway
from pearlarr.sonarr_client import AbstractSonarrClient
from pearlarr.sonarr_episodes import SonarrEpisodes
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.stamps import STAMP_FORMAT, parse_stamp, stamp_of
from pearlarr.torrent_service import AddResult, AddStatus, TorrentService
from pearlarr.tvdb_orders import NO_ORDERS, CachedOrders, IndexOrders, OrderRow, OrdersByName
from pearlarr.wait_types import (
    NO_HAND_IMPORT_FILES,
    AttemptResult,
    HandImportFile,
    ImportWaitMode,
    LeftForHandImport,
    PauseReason,
)
from pearlarr.window_placement import WindowedPlacements

from .fakes import FakeClock, FakeRadarrClient, FakeSkyHookClient, FakeSonarrClient

# The display separator the wait region, boot step lines and report rows join parts with. Assertions build expected
# strings from this so a separator change is one edit.
SEP = " · "

# Flat (group-local) setting name -> config group, derived from AppConfig's own field tree so it can't
# drift into an outdated subset. AppConfig declares `sonarr` before `radarr`, so a name shared by both arr
# groups (url/api_key/ignore_unmonitored/torrent_category) resolves to `sonarr`.
_FIELD_GROUP: dict[str, str] = {}
for _group, _group_field in AppConfig.model_fields.items():
    # `annotation` is the group's submodel class. The isinstance guard skips the top-level scalar (config_version).
    _submodel: Any = _group_field.annotation
    if not (isinstance(_submodel, type) and issubclass(_submodel, BaseModel)):
        continue
    for _field in _submodel.model_fields:
        _FIELD_GROUP.setdefault(_field, _group)

_FLAT_ALIASES: dict[str, tuple[str, str]] = {
    "import_wait_mode": ("imports", "wait_mode"),
    "import_poll_interval": ("imports", "poll_interval"),
    "import_default_quality": ("imports", "default_quality"),
    "import_languages_dual": ("imports", "languages_dual"),
    "import_languages_single": ("imports", "languages_single"),
    "max_torrents_to_add": ("advanced", "max_torrents_to_add"),
    "sleep_time": ("advanced", "sleep_time"),
    "log_level": ("advanced", "log_level"),
    "discord_url": ("notifications", "discord_url"),
    "wait_webhook_url": ("notifications", "wait_webhook_url"),
    "wait_notify": ("notifications", "wait_notify"),
    "torrent_tags": ("qbittorrent", "tags"),
    "sonarr_ignore_unmonitored": ("sonarr", "ignore_unmonitored"),
    "radarr_ignore_unmonitored": ("radarr", "ignore_unmonitored"),
    "sonarr_torrent_category": ("sonarr", "torrent_category"),
    "radarr_torrent_category": ("radarr", "torrent_category"),
    "sonarr_post_import_category": ("sonarr", "post_import_category"),
    "radarr_post_import_category": ("radarr", "post_import_category"),
    # Bare url/api_key resolve to sonarr (first-wins), so these are how the Radarr-run builders
    # and tests reach the radarr connection keys.
    "radarr_url": ("radarr", "url"),
    "radarr_api_key": ("radarr", "api_key"),
}


def _resolve_setting(key: str) -> tuple[str, str]:
    """Map a flat override key to its `(group, field)` in the nested config."""

    if key in _FLAT_ALIASES:
        return _FLAT_ALIASES[key]
    return _FIELD_GROUP.get(key, "seadex"), key


# The override keys make_services routes into self._config rather than onto the bare instance as an attribute.
_CONFIG_SETTING_NAMES = frozenset(_FIELD_GROUP) | frozenset(_FLAT_ALIASES)


def _split_config(overrides: dict[str, Any]) -> AppConfig:
    """Pop the config-routed keys out of `overrides` (IN PLACE) and build the config."""

    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}
    return make_config(**config_overrides)


def make_bare_instance[T](cls: type[T], **attrs: Any) -> T:
    """An instance with `__init__` bypassed and only the given attrs set."""

    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


# The cached-title columns `update_cached_title` merges, aliased from the real store's tuple so the fake
# can't drift from `CacheStore`.
_FAKE_CACHED_TITLE_FIELDS: tuple[str, ...] = _CACHED_TITLE_COLUMNS


def _evict_stale[K](store: dict[K, dict[str, Any]], cutoff: datetime) -> int:
    """Drop rows whose `fetched_at` is stamp-less or older than `cutoff`."""

    cutoff_str = cutoff.strftime(STAMP_FORMAT)
    stale: list[K] = []
    for cache_key, row in store.items():
        stamp = row.get("fetched_at")
        if not isinstance(stamp, str) or stamp < cutoff_str:
            stale.append(cache_key)
    for cache_key in stale:
        del store[cache_key]
    return len(stale)


def anilist_body(al_id: int, title: str = "Resolved") -> AniListBody:
    """A populated single-id AniList body: the run-cache value, and a stored answer's `data`."""

    return {
        "data": {
            "Media": {
                "id": al_id,
                "title": {"english": title},
                "coverImage": {"large": "https://img/large"},
                "bannerImage": "https://img/banner",
                "episodes": 12,
                "format": "TV",
                "startDate": {"year": 2020, "month": 1, "day": 4},
                "endDate": {"year": 2020, "month": 3, "day": 21},
                "duration": 24,
            },
        },
    }


class ScriptedAniListClient(AniListClient):
    """Scripted AniList wire client: canned bodies, wire calls recorded, no HTTP.

    `title` names every known id (None: AniList knows none), an `absent` id answers the unknown-id
    body, and a request touching a `failing` id marks AniList unreachable, muting later calls like the real client.
    """

    def __init__(
        self,
        title: str | None = "Resolved",
        *,
        absent: frozenset[int] = frozenset(),
        failing: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__(client=httpx.Client())
        self._title = title
        self._absent = absent
        self._failing = failing
        self.query_calls: list[int] = []
        self.batch_calls: list[list[int]] = []

    def _trip(self) -> None:
        self._note_unreachable(f"request failed after {MAX_RETRIES} retries")

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        if self.unreachable:
            return {}
        self.query_calls.append(al_id)
        if al_id in self._failing:
            self._trip()
            return {}
        if self._title is None or al_id in self._absent:
            return {"data": {"Media": None}}
        return anilist_body(al_id, self._title)

    @override
    def query_batch(self, al_ids: list[int]) -> AniListAnswers:
        if self.unreachable:
            return {}
        self.batch_calls.append(list(al_ids))
        if self._failing & set(al_ids):
            self._trip()
            return {}
        if self._title is None:
            return {}
        return {al_id: anilist_body(al_id, self._title) for al_id in al_ids if al_id not in self._absent}


class FakeCacheStore(AbstractCacheStore):
    """In-memory stand-in mirroring the SQLite `CacheStore` public facade."""

    def __init__(
        self,
        *,
        sonarr_parse: dict[str, dict[str, Any]] | None = None,
        pending: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._sonarr_parse: dict[str, dict[str, Any]] = dict(sonarr_parse or {})
        self._pending: dict[str, dict[str, dict[str, Any]]] = {arr: dict(recs) for arr, recs in (pending or {}).items()}
        # The cached titles, keyed by (arr, al_id). The existence checks key on membership, never the dict's truthiness.
        self._cached_titles: dict[tuple[str, int], dict[str, Any]] = {}
        self._anilist_meta: dict[int, dict[str, Any]] = {}
        self._guards: dict[str, dict[int, OwnedAtGrab]] = {}
        self._history_checkpoints: dict[str, HistoryCheckpoint] = {}
        self._tvdb_orders: dict[int, CachedOrders] = {}
        self._kv: dict[str, str] = {}

    # -- lifecycle --
    @override
    def save(self, *, dry_run: bool) -> None:
        del dry_run

    @override
    def close(self) -> None:
        pass

    # -- selection hash --
    @override
    def selection_changed(self, arr: Arr, selection_hash: str) -> bool:
        stored = self._kv.get(selection_hash_key(arr))
        return stored is not None and stored != selection_hash

    @override
    def store_selection_hash(self, arr: Arr, selection_hash: str) -> None:
        self._kv[selection_hash_key(arr)] = selection_hash

    # -- per-title rows (cached titles) --
    @override
    def update_cached_title(
        self,
        arr: Arr,
        al_id: int,
        update: CachedTitleUpdate,
    ) -> None:
        """Partial-merge the supplied columns. An empty update adds no row."""

        details: dict[str, Any] = dict(update)
        updated_at = details.get("updated_at")
        if isinstance(updated_at, datetime):
            details["updated_at"] = updated_at.strftime(STAMP_FORMAT)

        supplied = {column: details[column] for column in _FAKE_CACHED_TITLE_FIELDS if column in details}
        if supplied:
            self._cached_titles.setdefault((str(arr), al_id), {}).update(supplied)

    @override
    def uncache_title(self, arr: Arr, al_id: int) -> None:
        """Drop the title's stored timestamp. No row, no effect."""

        cached = self._cached_titles.get((str(arr), al_id))
        if cached is not None:
            cached.pop("updated_at", None)

    @override
    def get_cached_title(self, arr: Arr, al_id: int) -> CachedTitle | None:
        """The title's row as a `CachedTitle`, or None."""

        cached = self._cached_titles.get((str(arr), al_id))
        if cached is None:
            return None
        return CachedTitle(
            updated_at=cached.get("updated_at"),
            name=cached.get("name"),
            url=cached.get("url"),
            coverage=cached.get("coverage"),
            fallback_satisfied=bool(cached.get("fallback_satisfied", False)),
            season_type=cached.get("season_type"),
        )

    # -- AniList meta (TTL-swept) --
    @override
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        yield from ((al_id, deepcopy(rec)) for al_id, rec in list(self._anilist_meta.items()))

    @override
    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        return deepcopy(self._anilist_meta.get(al_id))

    @override
    def put_anilist_meta(self, al_id: int, row: dict[str, Any]) -> None:
        self._anilist_meta[al_id] = deepcopy(row)

    @override
    def evict_anilist_meta(self, cutoff: datetime) -> int:
        return _evict_stale(self._anilist_meta, cutoff)

    # -- Sonarr parse cache (TTL-swept) --
    @override
    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        return deepcopy(self._sonarr_parse.get(filename))

    @override
    def put_sonarr_parse(self, filename: str, row: dict[str, Any]) -> None:
        self._sonarr_parse[filename] = deepcopy(row)

    @override
    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        return _evict_stale(self._sonarr_parse, cutoff)

    # -- pending imports --
    @override
    def get_pending_import_rows(self, arr: Arr) -> dict[str, dict[str, Any]]:
        return {key: deepcopy(rec) for key, rec in self._pending.get(str(arr), {}).items()}

    @override
    def get_pending_import_row(self, arr: Arr, infohash: str) -> dict[str, Any] | None:
        return deepcopy(self._pending.get(str(arr), {}).get(infohash))

    @override
    def get_pending_import_rows_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]:
        """Fresh deep copy of the pending imports with a claim on the series (mirrors the `json_each` read)."""

        return {
            key: deepcopy(rec)
            for key, rec in self._pending.get(str(arr), {}).items()
            if any(claim.get("series_id") == series_id for claim in rec.get("claims", []))
        }

    @override
    def put_pending_import_row(self, arr: Arr, infohash: str, row: dict[str, Any]) -> None:
        # Stored as the real store's JSON column reads it back (tuples come back as lists).
        self._pending.setdefault(str(arr), {})[infohash] = json.loads(json.dumps(row))

    @override
    def drop_pending_import_row(self, arr: Arr, infohash: str) -> None:
        self._pending.get(str(arr), {}).pop(infohash, None)

    @override
    def other_arr_holds(self, arr: Arr, infohash: str) -> bool:
        """Whether the other arr has a pending import on the torrent, by exact key (mirrors the SQL)."""

        return any(infohash in recs for arr_key, recs in self._pending.items() if arr_key != str(arr))

    @override
    def put_guards(self, arr: Arr, al_id: int, guards: OwnedAtGrab) -> None:
        self._guards.setdefault(str(arr), {})[al_id] = guards

    @override
    def get_guards(self, arr: Arr) -> dict[int, OwnedAtGrab]:
        """Only titles with a live pending import (mirrors the real store's join)."""

        live = {
            claim.get("al_id") for rec in self._pending.get(str(arr), {}).values() for claim in rec.get("claims", [])
        }
        return {al_id: g for al_id, g in self._guards.get(str(arr), {}).items() if al_id in live}

    # -- TVDB orders --
    @override
    def get_tvdb_orders(self, tvdb_id: int) -> CachedOrders | None:
        """Round-trips through the stored JSON shape, as the real store's column does."""

        stored = self._tvdb_orders.get(tvdb_id)
        return None if stored is None else CachedOrders.from_json(json.loads(json.dumps(stored.to_json())))

    @override
    def put_tvdb_orders(self, tvdb_id: int, row: CachedOrders) -> None:
        self._tvdb_orders[tvdb_id] = row

    # -- history checkpoints --
    @override
    def get_history_checkpoint(self, arr: Arr) -> HistoryCheckpoint | None:
        return self._history_checkpoints.get(str(arr))

    @override
    def put_history_checkpoint(self, arr: Arr, checkpoint: HistoryCheckpoint) -> None:
        self._history_checkpoints[str(arr)] = checkpoint

    # -- maintenance: stats, integrity --
    @override
    def stats(self) -> CacheStats:
        return CacheStats(
            entries=len(self._cached_titles),
            anilist_meta=len(self._anilist_meta),
            sonarr_parse=len(self._sonarr_parse),
            pending_imports=sum(len(recs) for recs in self._pending.values()),
            guard_facts=sum(len(g) for g in self._guards.values()),
            tvdb_orders=len(self._tvdb_orders),
            size_bytes=0,
        )

    @override
    def integrity_check(self) -> str:
        return "ok"


class FakeSeaDexSource(SeaDexSource):
    """In-memory `SeaDexSource` stand-in: serves preset entries, no network."""

    def __init__(self, entries: dict[int, EntryRecord] | None = None, *, unreachable: bool = False) -> None:
        self._entries: dict[int, EntryRecord] = dict(entries or {})
        self._unreachable = unreachable
        self.prefetch_calls: list[list[int]] = []
        self.entry_calls: list[int] = []

    @override
    def prefetch(self, al_ids: Iterable[int], *, progress: ProgressCallback | None = None) -> int:
        del progress
        ids = list(al_ids)
        self.prefetch_calls.append(ids)
        return len(ids)

    @override
    def entry(self, al_id: int) -> EntryRecord | SeaDexNoEntry:
        self.entry_calls.append(al_id)
        found = self._entries.get(al_id)
        if found is not None:
            return found
        return SeaDexNoEntry.UNREACHABLE if self._unreachable else SeaDexNoEntry.NOT_FOUND

    @property
    @override
    def unreachable(self) -> bool:
        return self._unreachable


def make_logger(name: str = "pearlarr-test") -> logging.Logger:
    """A quiet logger (null handler, no propagation), reset to WARNING on every call."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    return logger


def make_anilist_gateway(client: AniListClient | None = None, store: FakeCacheStore | None = None) -> AniListGateway:
    """A real gateway over an in-memory store and a network-free wire client (scripted, or bare and never queried)."""

    return AniListGateway(
        cache_store=store if store is not None else FakeCacheStore(),
        logger=make_logger(),
        client=client if client is not None else AniListClient(client=httpx.Client()),
    )


def make_config(**overrides: Any) -> AppConfig:
    """An in-memory `AppConfig` carrying the decision-test defaults, `trackers` left unset (PUBLIC | PRIVATE)."""

    nested: dict[str, dict[str, Any]] = {
        "seadex": {
            "want_best": True,
            "prefer_dual_audio": True,
            "ignore_tags": [],
        },
        "advanced": {"interactive": False},
    }
    for key, value in overrides.items():
        group, field = _resolve_setting(key)
        nested.setdefault(group, {})[field] = value
    return AppConfig.model_validate(nested)


def title_is_cached(row: CachedTitle | None, sd_entry: EntryRecord) -> bool:
    """Whether a cache row has its title cached under SeaDex's current update time."""

    return row is not None and row.cached_under(sd_entry.updated_at)


def make_entry_record(
    *,
    anilist_id: int = 1,
    url: str = "https://releases.moe/1",
    is_incomplete: bool = False,
    updated_at: datetime | None = None,
    torrents: tuple[TorrentRecord, ...] = (),
    size: int = 0,
    notes: str = "",
    comparisons: tuple[str, ...] = (),
) -> EntryRecord:
    """A real `seadex.EntryRecord` (frozen msgspec) with all 13 fields defaulted."""

    stamp = updated_at if updated_at is not None else datetime(2026, 1, 1)
    return EntryRecord(
        anilist_id=anilist_id,
        collection_id="col",
        collection_name="col-name",
        comparisons=comparisons,
        created_at=stamp,
        id="entry1",
        is_incomplete=is_incomplete,
        notes=notes,
        theoretical_best=None,
        torrents=torrents,
        updated_at=stamp,
        url=url,
        size=size,
    )


def make_torrent_record(
    *,
    release_group: str = "SubsPlease",
    tracker: Tracker = Tracker.NYAA,
    url: str = "https://nyaa.si/1",
    infohash: str | None = "a" * 40,
    file_names: tuple[str, ...] = (),
    file_size: int = 1000,
    file_sizes: tuple[int, ...] | None = None,
    is_dual_audio: bool = False,
    is_best: bool = True,
    size: int = 1000,
    tags: frozenset[Tag] = frozenset(),
) -> TorrentRecord:
    """A real `seadex.TorrentRecord` (frozen msgspec) whose `file_names` become sized `File` entries."""

    stamp = datetime(2026, 1, 1)
    sizes = file_sizes if file_sizes is not None else (file_size,) * len(file_names)
    return TorrentRecord(
        collection_id="c",
        collection_name="cn",
        created_at=stamp,
        is_dual_audio=is_dual_audio,
        files=tuple(File(name=name, size=file_bytes) for name, file_bytes in zip(file_names, sizes, strict=True)),
        id="t1",
        infohash=infohash,
        is_best=is_best,
        release_group=release_group,
        tags=tags,
        tracker=tracker,
        updated_at=stamp,
        url=url,
        grouped_url=None,
        size=size,
    )


def _real_reporter(
    logger: logging.Logger,
    cache_store: AbstractCacheStore,
    web: httpx.Client,
) -> RunReporter:
    """A real `RunReporter` over the given cache store (composite-with-faked-leaf)."""

    counts = SeverityCounter()
    return RunReporter(
        emit=emit_to_hub,
        counts=lambda: counts,
        cache_store=cache_store,
        anilist=AniListGateway(cache_store=cache_store, logger=logger, client=AniListClient(client=web)),
    )


def make_categories(
    config: AppConfig | None = None,
    arr: Arr = Arr.SONARR,
    *,
    sources: CategorySources | None = None,
) -> ArrCategoryResolver:
    """A category resolver, fetchless by default (no sources: config values pass through, omitted stays blank).

    `sources` binds a real transport and a qBittorrent stub for the live-fetch mode.
    """

    config = config or AppConfig()
    return ArrCategoryResolver(arr, config.for_arr(arr), sources)


def make_fetched_categories(
    config: AppConfig | None = None,
    arr: Arr = Arr.SONARR,
    *,
    grab: str | None,
    post_import: str | None,
    qbit: qbittorrentapi.Client | None = None,
) -> ArrCategoryResolver:
    """A resolver seeded as if its one live fetch already returned one client with `(grab, post_import)`.

    Pokes the resolver's private `_fetched` state directly, with the production read of that one client.
    `qbit` is the qBittorrent stub its torrent and settings reads go to. Without one it reads neither.
    """

    config = config or AppConfig()
    sources = None
    if qbit is not None:
        # Never fetched through: the read is seeded below.
        http = ArrHttp.bind(client=httpx.Client(), url="http://arr.test", api_key="testkey", label=arr.capitalize())
        sources = CategorySources(http, qbit)
    resolver = ArrCategoryResolver(arr, config.for_arr(arr), sources)
    names = _CATEGORY_FIELDS[arr]
    fields = [{"name": names.grab, "value": grab or ""}, {"name": names.post_import, "value": post_import or ""}]
    resolver._fetched = _read_clients(arr, [DownloadClientRecord.model_validate(download_client_json(fields))])
    return resolver


def download_client_json(
    fields: Iterable[object],
    *,
    enable: object = True,
    implementation: object = "QBittorrent",
    priority: object = 1,
) -> dict[str, object]:
    """One realistic enabled-qBittorrent `DownloadClientResource` body carrying `fields` (junk allowed)."""

    return {
        "enable": enable,
        "protocol": "torrent",
        "priority": priority,
        "name": "qBittorrent",
        "fields": list(fields),
        "implementationName": "qBittorrent",
        "implementation": implementation,
        "configContract": "QBittorrentSettings",
        "id": 1,
    }


def sonarr_client_fields(grab: str = "tv-sonarr", post_import: str = "sonarr-done") -> list[dict[str, object]]:
    """The Sonarr client's category fields (plus the connection noise a real body carries)."""

    return [
        {"name": "host", "value": "localhost"},
        {"name": "port", "value": 8080},
        {"name": "tvCategory", "value": grab},
        {"name": "tvImportedCategory", "value": post_import},
    ]


def _real_torrents(logger: logging.Logger, web: httpx.Client, categories: ArrCategoryResolver) -> TorrentService:
    """A real, client-less `TorrentService` (`qbit=None` -> dry-run no-op add)."""

    return TorrentService(qbit=None, web=web, categories=categories, tags=[], logger=logger)


def make_services(**overrides: Any) -> ArrRunServices:
    """A bare `ArrRunServices` carrying only the attributes its methods read."""

    logger = make_logger()

    config = _split_config(overrides)
    defaults: dict[str, Any] = {
        "logger": logger,
        "_config": config,
        # The real __init__ always sets the authoritative arr and an ArrRunState. Override _state=... for a run state.
        "arr": Arr.SONARR,
        "_state": ArrRunState(arr=Arr.SONARR),
        # The real __init__ mints these too. Without them the dirty-aware / selection-aware skip predicates fail.
        "_dirty_al_ids": set[int](),
        "_selection_changed": False,
        "_clock": FakeClock(),
        # `bind_arr_run` tells the torrent service who imports.
        "_torrents": _real_torrents(logger, httpx.Client(), make_categories()),
    }
    defaults.update(overrides)
    return make_bare_instance(ArrRunServices, **defaults)


def make_run_deps(
    *,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    seadex: SeaDexSource | None = None,
    logger: logging.Logger | None = None,
    clock: Clock | None = None,
    skyhook_client: AbstractSkyHookClient | None = None,
    categories: ArrCategoryResolver | None = None,
) -> ArrRunDeps:
    """A real `ArrRunDeps` over typed fakes, with a Sonarr url and api_key set and `qbit` None (dry run).

    `categories` swaps in a resolver that reads Sonarr's clients, in place of the config passthrough.
    """

    config = config or make_config(url="http://sonarr", api_key="key")
    cache_store = cache_store or FakeCacheStore()
    logger = logger or make_logger()
    # One shared client backs both deps.http and deps.web. conftest's close_leaked_handles closes it at teardown.
    http = httpx.Client()
    # No transport: the config passthrough (no download-client fetch under test). Shared with
    # torrents below, matching production's single resolver instance.
    categories = categories or make_categories(config)
    return ArrRunDeps(
        config=config,
        arr_config=config.for_arr(Arr.SONARR),
        # None: the strategies' require_connection binds one lazily, so the keys-missing
        # construction paths behave exactly as production's.
        arr_http=None,
        categories=categories,
        clock=clock or FakeClock(),
        web=http,
        parse_cache_rows=ParseCacheRows(cache_store),
        http=http,
        qbit=None,
        # A real resolver over empty in-memory mappings (no network). It carries a real (empty)
        # `anibridge` the strategy reads at construction.
        resolver=MappingResolver(
            refresh_days=1,
            ignore_anilist_ids=set(),
            web=http,
            sources=MappingSources(anime={}, anidb=False, anibridge=False),
        ),
        logger=logger,
        seadex=seadex or FakeSeaDexSource(),
        cache_store=cache_store,
        anilist=AniListGateway(cache_store=cache_store, logger=logger, client=AniListClient(client=http)),
        skyhook=SkyHookGateway(client=skyhook_client or FakeSkyHookClient(), cache_store=cache_store),
        torrents=_real_torrents(logger, http, categories),
        notifier=Notifier(discord_url=None, webhook_url=None, web=http),
        planner=make_planner(),
        reporter=_real_reporter(logger, cache_store, http),
    )


def make_release_filter(**overrides: Any) -> SeadexReleaseFilter:
    """A `SeadexReleaseFilter` over an assembled `ArrRunDeps`, with `state` and `planner` overridable."""

    config = _split_config(overrides)
    state = overrides.pop("state", None) or ArrRunState(arr=Arr.SONARR)
    deps = make_run_deps(config=config)
    if "planner" in overrides:
        deps = dataclasses.replace(deps, planner=overrides.pop("planner"))
    if overrides:
        msg = f"unknown make_release_filter overrides: {sorted(overrides)}"
        raise TypeError(msg)
    return SeadexReleaseFilter(deps=deps, state=state)


# A truthy stand-in for a logged-in qBittorrent client, so is_dry_run() is False.
CLIENT_SENTINEL = object()


def _lower_hash(infohash: str | None) -> str | None:
    """A scripted hash as the release carries it (lowercased)."""

    return None if infohash is None else infohash.lower()


class FakeTorrents:
    """Mimics `TorrentService.add`: a per-hash scripted `(outcome, name)` or full `AddResult`."""

    def __init__(
        self,
        by_hash: dict[str | None, tuple[AddStatus, str | None] | AddResult],
        *,
        raises: dict[str | None, Exception] | None = None,
    ) -> None:
        # The SeaDex boundary lowercases hashes, so a scripted mixed-case hash is read lowercased too.
        self._by_hash = {_lower_hash(h): scripted for h, scripted in by_hash.items()}
        self._raises = {_lower_hash(h): exc for h, exc in (raises or {}).items()}
        self.calls: list[str | None] = []

    def add(
        self,
        *,
        item: Release,
        dry_run: bool,
    ) -> AddResult:
        """Return the scripted `AddResult` for the release's infohash, or raise its scripted error."""

        del dry_run
        infohash = item.infohash
        self.calls.append(infohash)
        if infohash in self._raises:
            raise self._raises[infohash]
        scripted = self._by_hash[infohash]
        return scripted if isinstance(scripted, AddResult) else AddResult(*scripted)


def one_release_dict(*, srg: str, infohash: str, url: str = "https://nyaa.si/view/1") -> SeadexDict:
    """A one-release `SeadexDict` marked to grab, its tracker pinned to `NYAA`."""

    item = make_release(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return {srg: rg_group({url: item})}


def torrent_facts(
    *,
    infohash: str = "h1",
    release_group: str = "RG",
    is_dual_audio: bool = False,
    seadex_files: tuple[str, ...] = (),
    release_sizes: tuple[int, ...] = (),
    sizes_by_name: Mapping[str, int] = NO_SIZES_BY_NAME,
) -> TorrentFacts:
    """Torrent h1 by group RG, single-audio, with no listed files or sizes unless given."""

    return TorrentFacts(
        infohash=infohash,
        release_group=release_group,
        is_dual_audio=is_dual_audio,
        seadex_files=seadex_files,
        release_sizes=release_sizes,
        sizes_by_name=sizes_by_name,
    )


def pending_import_update(
    infohash: str,
    *,
    stored: PendingImport | None = None,
    placements: FileEpisodeMap | None = None,
    hand_import_files: tuple[str, ...] = (),
    order_only: Mapping[str, OrderOnlyKey] = NO_ORDER_ONLY,
    **claim: Any,
) -> PendingImportUpdate:
    """A pending import update on `infohash` (group NAN0, no files) claimed by `title_claim(**claim)`, new unless `stored` is given."""

    return PendingImportUpdate(
        facts=torrent_facts(infohash=infohash, release_group="NAN0"),
        file_map=placements or {},
        excluded=(),
        hand_import_files=hand_import_files,
        order_only=order_only,
        claim=title_claim(**claim),
        pending_import=stored,
    )


def title_facts(**overrides: Any) -> TitleFacts:
    """Title 1 on series 7 named Show, with no coverage, url, or guard facts unless given."""

    defaults: dict[str, Any] = {
        "al_id": 1,
        "series_id": 7,
        "title": "Show",
        "coverage": None,
        "url": None,
        "guards": OwnedAtGrab(),
    }
    defaults.update(overrides)
    return TitleFacts(**defaults)


def grab_request(**overrides: Any) -> GrabRequest:
    """A minimal `GrabRequest` for driving the add path directly (an empty `seadex_dict`, no pending import updates)."""

    defaults: dict[str, Any] = {
        "al_id": 1,
        "arr_title": "Show",
        "title_name": "Show",
        "entry": make_entry_record(url="https://releases.moe/1"),
        "seadex_dict": {},
        "update": {},
        "replaced_groups": (),
    }
    defaults.update(overrides)
    return GrabRequest(**defaults)


def make_grab_pipeline(**overrides: Any) -> GrabPipeline:
    """A bare `GrabPipeline` carrying only what its methods read, `_state` a blocking run, not a dry run."""

    config = _split_config(overrides)
    logger = make_logger()
    cache_store = overrides.pop("cache_store", None) or FakeCacheStore()
    web = httpx.Client()
    defaults: dict[str, Any] = {
        "_config": config,
        "_planner": make_planner(),
        "cache_store": cache_store,
        "_torrents": _real_torrents(logger, web, make_categories(config)),
        "_anilist": AniListGateway(cache_store=cache_store, logger=logger, client=AniListClient(client=web)),
        # No discord/webhook url: a disabled, best-effort no-op notifier.
        "_notifier": Notifier(discord_url=None, webhook_url=None, web=web),
        "_reporter": _real_reporter(logger, cache_store, web),
        "logger": logger,
        "qbit": CLIENT_SENTINEL,
        "_state": ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING),
    }
    defaults.update(overrides)
    if "_pending_imports" not in defaults:
        # The pending import store binds the FINAL store + state, exactly as the real ctor + bind_arr_run do.
        pending_imports = PendingImportStore(defaults["cache_store"])
        pending_imports.bind_arr_run(defaults["_state"])
        defaults["_pending_imports"] = pending_imports
    return make_bare_instance(GrabPipeline, **defaults)


def _categories_like_config(config: AppConfig, qbit: qbittorrentapi.Client | None) -> ArrCategoryResolver:
    """The Sonarr resolver for `config`: fetchless without `qbit`, else seeded with the config's own categories."""

    if qbit is None:
        return make_categories(config)
    settings = config.for_arr(Arr.SONARR)
    return make_fetched_categories(
        config, grab=settings.torrent_category, post_import=settings.post_import_category, qbit=qbit
    )


def make_import_wait_manager(**overrides: Any) -> ImportWaitManager:
    """A bare `ImportWaitManager` over real sub-objects, config keys routed through `AppConfig`.

    Constructs the pending import store, pollers and cleanup on the SAME store, binds the pending import store as
    `ArrRunServices` would, and ends with `begin_run`, so every sub-object shares the one state (and strategy) a run
    would.
    """

    config = overrides.pop("config", None) or _split_config(overrides)
    logger = overrides.pop("logger", None) or make_logger()
    cache_store = overrides.pop("cache_store", None) or FakeCacheStore()
    qbit = overrides.pop("qbit", None)
    clock = overrides.pop("clock", None) or FakeClock()
    # `ArrRunDeps.categories` as a run would have it. With no qBittorrent it is fetchless and the config
    # category passes through. With one, it reads as if the arr's own client carried the config's categories.
    categories = overrides.pop("categories", None) or _categories_like_config(config, qbit)
    # The production placeholder before a run binds one. Tests driving the import hook pass their own.
    strategy = overrides.pop("strategy", None)
    state = overrides.pop("state", None) or ArrRunState(arr=Arr.SONARR)
    reporter = overrides.pop("reporter", None) or _real_reporter(logger, cache_store, httpx.Client())
    if overrides:
        msg = f"unknown make_import_wait_manager overrides: {sorted(overrides)}"
        raise TypeError(msg)

    pending_imports = PendingImportStore(cache_store)
    pending_imports.bind_arr_run(state)
    pollers = make_bare_instance(WaitPollers, _qbit=qbit, _logger=logger, strategy=None)
    cleanup = make_bare_instance(
        PostImportCleanup,
        _imports=config.imports,
        _categories=categories,
        _qbit=qbit,
        _clock=clock,
        _logger=logger,
        _pending_imports=pending_imports,
        _pollers=pollers,
    )
    mgr = make_bare_instance(
        ImportWaitManager,
        imports=config.imports,
        clock=clock,
        logger=logger,
        _reporter=reporter,
        _pending_imports=pending_imports,
        pollers=pollers,
        _cleanup=cleanup,
    )
    mgr.bind_arr_run(state, strategy)
    return mgr


def make_planner(**overrides: Any) -> GrabPlanner:
    """A `GrabPlanner` with test-friendly defaults (arr `SONARR`, not interactive)."""

    logger = make_logger()

    defaults: dict[str, Any] = {
        "arr": Arr.SONARR,
        "interactive": False,
        "logger": logger,
    }
    defaults.update(overrides)
    return GrabPlanner(**defaults)


def make_release(
    *,
    url: str = "https://nyaa.si/view/1",
    files: list[str] | None = None,
    size: list[int] | None = None,
    tracker: Tracker = Tracker.OTHER,
    is_public: bool = True,
    is_dual_audio: bool = False,
    infohash: str | None = "hash1",
    download: bool = False,
    is_fallback: bool = False,
    upgrade: bool = False,
    episodes: list[ReleaseEpisode] | None = None,
) -> Release:
    """One SeaDex release, as `get_seadex_dict`'s dict carries it."""

    return Release(
        url=url,
        files=files or [],
        file_sizes=size or [],
        tracker=tracker,
        is_public=is_public,
        is_dual_audio=is_dual_audio,
        infohash=infohash,
        to_grab=download,
        is_fallback=is_fallback,
        is_upgrade=upgrade,
        episodes=episodes or [],
    )


def grab_plan(seadex_dict: SeadexDict) -> GrabPlan:
    """A `GrabPlan` carrying just the dict, with no skips and no guard evidence."""

    return GrabPlan(seadex_dict=seadex_dict, skips=PrivateOnlyNotices(), guards=OwnedAtGrab())


def rg_group(
    urls: dict[str, Release],
    *,
    tags: frozenset[Tag] | None = None,
    all_episodes: list[ReleaseEpisode] | None = None,
) -> GroupReleases:
    """`all_episodes` is tri-state: `None` skips parsing, `[]` is unparsed, populated is the coverage frozenset."""

    return GroupReleases(
        releases=urls,
        tags=tags or frozenset(),
        all_episodes=all_episodes,
    )


def sonarr_ep(
    season: int | None,
    episode: int | None,
    *,
    size: int | None = None,
    release_group: str | None = None,
    episode_file_id: int = 1,
    ep_id: int = 0,
    absolute: int | None = None,
    title: str = "",
    tvdb_id: int | None = None,
    air_date: str | None = None,
    runtime: int | None = None,
) -> SonarrEpisode:
    """`episode_file_id=0` omits the `episodeFile` block, as Sonarr sends it for a missing episode.

    `air_date` is Sonarr's `YYYY-MM-DD` string and `runtime` its minutes.
    """

    raw: dict[str, Any] = {
        "id": ep_id,
        "seasonNumber": season,
        "episodeNumber": episode,
        "absoluteEpisodeNumber": absolute,
        "title": title,
        "tvdbId": tvdb_id,
        "airDate": air_date,
        "runtime": runtime,
        "episodeFileId": episode_file_id,
    }
    if episode_file_id:
        raw["episodeFile"] = {"size": size, "releaseGroup": release_group}
    return SonarrEpisode.model_validate(raw)


def arr_files(ep_list: Iterable[SonarrEpisode]) -> ArrFiles:
    """The episodes' files merged per group, files with no group left out, as `sonarr_files` reports them."""

    return ArrFiles.from_files(
        [f for ep in ep_list if ep.episode_file_id != 0 and (f := ep.episode_file) is not None], keep_no_group=False
    )


_NO_TORRENT_GROUP = TorrentGroup("", ())
"""A torrent group with no sizes: no file counts as misplaced against it."""


def episode_statuses(
    by_id: Mapping[int, EpisodeFileStatus], holding_own_copy: frozenset[int] = frozenset()
) -> EpisodeStatuses:
    """`EpisodeStatuses` over `by_id`, where `holding_own_copy` are the file map episodes with a copy of this torrent's file."""

    return EpisodeStatuses(by_id, holding_own_copy)


def episode_snapshot(
    *,
    episodes: EpisodeIndex | None = None,
    trusted: TrustedGroups | None = None,
    torrent_group: TorrentGroup = _NO_TORRENT_GROUP,
    owned_episode_sizes: Mapping[int, int] | None = None,
) -> EpisodeSnapshot:
    """An empty series index trusting no group, with no owned sizes unless given.

    The default torrent group has no sizes, so no file ever counts as misplaced unless `torrent_group` is given.
    """

    return EpisodeSnapshot(
        episodes=episode_index([]) if episodes is None else episodes,
        trusted_groups=trusted or {},
        torrent_group=torrent_group,
        owned_episode_sizes=owned_episode_sizes or {},
    )


_NO_SIZES: EpisodeBySize = MappingProxyType({})
"""No episodes by size, so nothing gets placed by size."""

UNLISTED = FetchedSeadex(EMPTY_LISTING, _NO_SIZES)
"""A torrent no entry lists, with the listings read and no episodes by size."""


UNREAD_EVIDENCE = FetchedSeadex(None, None)
"""SeaDex couldn't be read: no listing and no episodes by size."""


_NO_NAMES = TitleNames()
"""A title with no series or AniList names."""


def title_window(
    al_id: int,
    ids: Iterable[int],
    *,
    series_id: int = 7,
    names: TitleNames = _NO_NAMES,
    ignored: bool = False,
    episode_count: int | None = None,
) -> TitleWindow:
    """Title `al_id`'s window over `ids` on `series_id` (series 7, as `title_claim`'s), in the order given."""

    return TitleWindow(
        al_id=al_id,
        series_id=series_id,
        ordered_episode_ids=tuple(ids),
        names=names,
        ignored=ignored,
        episode_count=episode_count,
    )


def torrent_listing(*windows: TitleWindow, aliases: SpecialAliases = NO_ALIASES) -> TorrentListing:
    """The listing of a torrent SeaDex lists under the `windows`' titles, with `aliases`."""

    return TorrentListing(windows, aliases)


ALIASED_SPECIALS_LISTING = torrent_listing(title_window(1, (502, 504, 506)), aliases={1: 2, 2: 4, 3: 6})
"""A listing of specials 2, 4 and 6 whose aliases map a torrent's 1, 2 and 3 onto them."""


def torrent_evidence(
    *windows: TitleWindow, episode_by_size: EpisodeBySize = _NO_SIZES, aliases: SpecialAliases = NO_ALIASES
) -> FetchedSeadex:
    """Evidence after a good read: the `windows`' listing with `aliases`, and `episode_by_size` (both none by default)."""

    return FetchedSeadex(torrent_listing(*windows, aliases=aliases), episode_by_size)


def known_torrent(
    pending: PendingImport | None = None,
    *,
    indexes: Mapping[int, EpisodeIndex] | None = None,
    evidence: FetchedSeadex = UNLISTED,
) -> KnownTorrent:
    """A `KnownTorrent`, by default a new torrent that nothing lists, with no series read."""

    return KnownTorrent(pending, indexes or {}, evidence)


def make_grabbing_title(
    title_index: EpisodeIndex,
    series: EpisodeIndex,
    *,
    al_id: int = 1,
    series_id: int = 7,
    names: TitleNames = _NO_NAMES,
    episode_count: int | None = None,
    alternate_titles: tuple[str, ...] = (),
    aired_order: bool = True,
) -> GrabbingTitle:
    """The grabbing title `al_id` over `title_index`, within `series` on `series_id` (series 7, as `title_claim`'s).

    Unless given, the series has no alternate titles and Sonarr numbers it as aired.
    """

    return GrabbingTitle(
        al_id=al_id,
        series_id=series_id,
        title_index=title_index,
        series_index=series,
        title_names=names,
        episode_count=episode_count,
        alternate_titles=alternate_titles,
        aired_order=aired_order,
    )


def new_torrent(grabbing_title: GrabbingTitle) -> KnownTorrent:
    """A new torrent that nothing lists, with only the grabbing title's series read."""

    return known_torrent(indexes={grabbing_title.series_id: grabbing_title.series_index})


def series_index(
    id_by_key: Mapping[EpisodeKey, int],
    *,
    absolutes: Mapping[int, int] | None = None,
    titles: Mapping[int, str] | None = None,
    aired: Mapping[int, str | None] | None = None,
    orders: OrdersByName = NO_ORDERS,
) -> EpisodeIndex:
    """A whole-series index synthesized from a `(season, episode) -> id` dict, with absolutes and titles given by id.

    `aired` gives air dates by id, as `YYYY-MM-DD`, None for an episode without one. Each episode's TVDB id is its
    Sonarr id, so a row of `orders` names an episode by that id, and a row with any other TVDB id is an episode
    Sonarr doesn't list.
    """

    absolute_of = absolutes or {}
    title_of = titles or {}
    aired_on = aired or {}
    return episode_index(
        [
            sonarr_ep(
                key.season,
                key.episode,
                ep_id=ep_id,
                absolute=absolute_of.get(ep_id),
                title=title_of.get(ep_id, ""),
                tvdb_id=ep_id,
                air_date=aired_on.get(ep_id),
            )
            for key, ep_id in id_by_key.items()
        ],
        IndexOrders(orders, None),
    )


def parsed_info(
    *,
    season: int | None = None,
    episodes: tuple[int, ...] = (),
    absolutes: tuple[int, ...] = (),
    matched: tuple[tuple[int, int], ...] = (),
    full_season: bool = False,
    series_title: str = "",
) -> ParsedFileInfo:
    """Shorthand `ParsedFileInfo` for the placement tests: the name's own numbers and title, plus any matched pairs."""

    return ParsedFileInfo(
        season_number=season,
        episode_numbers=episodes,
        absolute_episode_numbers=absolutes,
        matched_episodes=tuple(
            MatchedEpisode(season_number=matched_season, episode_number=episode) for matched_season, episode in matched
        ),
        full_season=full_season,
        series_title=series_title,
    )


def by_name(result: BatchPlacements) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
    """One assignment keyed name -> (ids, verdict), for the tests that pin every placement."""

    return {p.name: (p.ids, p.verdict) for p in result.placements}


def verdicts_of(result: BatchPlacements) -> set[PlacementVerdict]:
    """The distinct verdicts an assignment gave, for the tests that pin what kind of verdict a batch got."""

    return {p.verdict for p in result.placements}


def merged_of(result: WindowedPlacements) -> BatchPlacements:
    """Each file's verdict over every window, batch order (`WindowedFile.merged`)."""

    return BatchPlacements(tuple(file.merged for file in result.files))


def under_window(result: WindowedPlacements, at: int) -> BatchPlacements:
    """What the placer gave each file under the window at `at`, batch order (`WindowedFile.per_window`)."""

    return BatchPlacements(tuple(file.per_window[at] for file in result.files))


def final_windows(result: WindowedPlacements) -> dict[str, int]:
    """Each file's name -> the position of its final window (`WindowedFile.final_window_at`), for the files with one."""

    return {file.name: file.final_window_at for file in result.files if file.final_window_at is not None}


def blind(names: Iterable[str]) -> dict[str, ParsedFileInfo | None]:
    """Every name parsed with no numbers (Sonarr read nothing)."""

    return dict.fromkeys(names, parsed_info())


def parsed_specials(*numbers: int) -> dict[str, ParsedFileInfo | None]:
    """`Show.S00Exx.mkv` per number, each read by Sonarr as that special."""

    return {f"Show.S00E{n:02d}.mkv": parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n in numbers}


def batch(
    parsed: Mapping[str, ParsedFileInfo | None],
    to_place: Sequence[str] | None = None,
    *,
    file_map: FileEpisodeMap = NO_FILE_MAP,
) -> PlacementBatch:
    """A batch over the whole torrent's parses: `to_place` (every parsed name when None) and the stored map `file_map`."""

    return PlacementBatch(list(parsed) if to_place is None else to_place, parsed, file_map)


def place(
    parsed: Mapping[str, ParsedFileInfo | None],
    target: PlacementTarget,
    to_place: Sequence[str] | None = None,
    *,
    file_map: FileEpisodeMap = NO_FILE_MAP,
) -> BatchPlacements:
    """The verdicts `place_batch` gives under `target` over `batch(parsed, to_place, file_map=file_map)`."""

    return place_batch(batch(parsed, to_place, file_map=file_map), target).placements


def numbered_names(prefix: str, count: int, tails: Sequence[str] = ()) -> list[str]:
    """`prefix - 01 [grp].mkv` names for `1..count`, each with its tail (a title after the number) when given."""

    return [
        f"{prefix} - {n:02d} - {tail} [grp].mkv" if tail else f"{prefix} - {n:02d} [grp].mkv"
        for n, tail in zip(range(1, count + 1), tails or [""] * count, strict=True)
    ]


# The `al_id` every `pending_import` builder's claim carries unless overridden, exported so tests can
# spell a built pending import's guard-row key without magic ints.
PENDING_IMPORT_AL_ID = 1

_CLAIM_KEYS = frozenset(TitleClaim.__dataclass_fields__)
PENDING_IMPORT_STAMP = "2026-06-24 00:00:00"
PENDING_IMPORT_ADDED = parse_stamp(PENDING_IMPORT_STAMP).astimezone()


def title_claim(**overrides: Any) -> TitleClaim:
    """A `TitleClaim` on series 7 by title `PENDING_IMPORT_AL_ID`, with no window unless `ordered_episode_ids` is given."""

    defaults: dict[str, Any] = {
        "al_id": PENDING_IMPORT_AL_ID,
        "series_id": 7,
        "title": "Show",
        "window_coverage": None,
        "url": None,
        "ordered_episode_ids": (),
        "names": TitleNames(),
        "preowned_episode_ids": (),
        "claimed_coverage": None,
    }
    defaults.update(overrides)
    for key in ("ordered_episode_ids", "preowned_episode_ids"):
        defaults[key] = tuple(defaults[key])
    return TitleClaim(**defaults)


def two_claim_pending_import(
    *,
    al_ids: tuple[int, int] = (1, 2),
    series_ids: tuple[int, int] = (7, 7),
    windows: tuple[tuple[int, ...], tuple[int, ...]] = ((101, 102), (103, 104)),
    titles: tuple[str, str] = ("Show", "Show"),
    **overrides: Any,
) -> PendingImport:
    """A `pending_import(**overrides)` claimed by two titles, each on its series over its window."""

    claims = tuple(
        title_claim(al_id=al_id, series_id=series_id, ordered_episode_ids=window, title=title)
        for al_id, series_id, window, title in zip(al_ids, series_ids, windows, titles, strict=True)
    )
    return pending_import(claims=claims, **overrides)


def pending_import(**overrides: Any) -> PendingImport:
    """A `PendingImport` wiring one mapped file to one episode id under one claim.

    Claim keys (`series_id`, `al_id`, `title`, `ordered_episode_ids`, ...) build that claim unless
    `claims` is given.
    """

    claim_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CLAIM_KEYS}
    defaults: dict[str, Any] = {
        "infohash": "abc123",
        "file_episode_map": {"show - 01 [1080p].mkv": (101,)},
        "release_group": "SubGroup",
        "is_dual_audio": False,
        "seadex_files": ("Show - 01 [1080p].mkv",),
        "added_at": PENDING_IMPORT_STAMP,
    }
    defaults.update(overrides)
    if "claims" not in defaults:
        defaults["claims"] = (title_claim(**claim_overrides),)
    for key in ("seadex_files", "excluded_files", "release_sizes"):
        if key in defaults:
            defaults[key] = tuple(defaults[key])
    defaults["file_episode_map"] = {name: tuple(ids) for name, ids in defaults["file_episode_map"].items()}
    return PendingImport(**defaults)


def order_only_hand_import_files(
    *, missing: tuple[EpisodeKey, ...] = (EpisodeKey(4, 1),), untrusted: tuple[EpisodeKey, ...] = ()
) -> LeftForHandImport:
    """A split opener's two halves left for a hand import under their DVD keys, the aired opener missing by default."""

    halves = ("[grp] Show Ending S1 - 01 [ABCD1234].mkv", "[grp] Show Ending S1 - 02 [ABCD1234].mkv")
    keys = (OrderOnlyKey("dvd", EpisodeKey(7, 1)), OrderOnlyKey("dvd", EpisodeKey(7, 2)))
    files = tuple(HandImportFile(half, key) for half, key in zip(halves, keys, strict=True))
    return LeftForHandImport(files=files, missing=missing, untrusted=untrusted)


def files_left_for_hand_import(*names: str) -> LeftForHandImport:
    """Files left for a hand import with no order-only key."""

    return LeftForHandImport(files=tuple(HandImportFile(name) for name in names))


def attempt_result(
    *,
    all_done: bool = True,
    command_issued: bool = False,
    imported_count: int = 0,
    file_map_episode_count: int = 0,
    pause_reason: PauseReason = PauseReason.NONE,
    placements: FileEpisodeMap | None = None,
    unmatched_files: tuple[str, ...] = (),
    left_for_hand_import: LeftForHandImport = NO_HAND_IMPORT_FILES,
) -> AttemptResult:
    """An `AttemptResult` defaulting to the all-done outcome (`all_done`)."""

    return AttemptResult(
        all_done=all_done,
        command_issued=command_issued,
        imported_count=imported_count,
        file_map_episode_count=file_map_episode_count,
        pause_reason=pause_reason,
        placements=placements or {},
        unplaced_files=unmatched_files,
        left_for_hand_import=left_for_hand_import,
    )


def manual_candidate(
    path: str,
    *,
    quality: dict[str, Any] | None = None,
    rejections: list[Any] | None = None,
) -> ManualImportCandidate:
    """`quality` is the raw wire dict. A rejection is a bare string or a `{"reason": ...}` dict."""

    return ManualImportCandidate.model_validate(
        {"path": path, "quality": quality, "rejections": rejections or []},
    )


def queue_record(
    infohash: str, state: str, *, status: str | None = "ok", queue_id: int = 0, series_id: int = 1
) -> QueueRecord:
    """`queue_id=0` means no usable id, `series_id=0` the unknown series."""

    return QueueRecord.model_validate(
        {
            "id": queue_id,
            "seriesId": series_id,
            "downloadId": infohash,
            "trackedDownloadState": state,
            "trackedDownloadStatus": status,
        },
    )


def make_sonarr_episodes(**attrs: Any) -> SonarrEpisodes:
    """A bare `SonarrEpisodes` with `__init__` bypassed, its per-run caches defaulted empty."""

    defaults: dict[str, Any] = {
        "_ep_list_cache": {},
        "_series_hash": "",
        "_order_facts": {},
        "_orderings": {},
        "_picks_logged": set[int](),
    }
    defaults.update(attrs)
    return make_bare_instance(SonarrEpisodes, **defaults)


def stored_orders(
    orders: Mapping[str, tuple[OrderRow, ...]],
    *,
    fresh_for: timedelta = timedelta(days=1),
    sonarr_counts: Mapping[str, int] | None = None,
) -> CachedOrders:
    """A series' cached order answer fetched a day ago, fresh for `fresh_for` from now (past it when negative)."""

    now = datetime.now()
    return CachedOrders(
        fetched_at=stamp_of(now - timedelta(days=1)),
        fresh_until=stamp_of(now + fresh_for),
        orders=orders,
        sonarr_counts=sonarr_counts,
    )


def make_sonarr_sync(
    *,
    sonarr: AbstractSonarrClient | None = None,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    ep_list_cache: dict[int, list[SonarrEpisode]] | None = None,
    clock: Clock | None = None,
    skyhook_client: AbstractSkyHookClient | None = None,
    categories: ArrCategoryResolver | None = None,
) -> SonarrStrategy:
    """A `SonarrStrategy` built through its real `__init__`, injecting typed clients and seeding `ep_list_cache`."""

    deps = make_run_deps(
        config=config, cache_store=cache_store, clock=clock, skyhook_client=skyhook_client, categories=categories
    )
    services = ArrRunServices(deps, Arr.SONARR)
    strat = SonarrStrategy(
        deps,
        services,
        sonarr_client=sonarr if sonarr is not None else FakeSonarrClient(),
    )
    if ep_list_cache is not None:
        strat._episodes._ep_list_cache = ep_list_cache
    return strat


def make_radarr_sync(
    *,
    radarr: AbstractRadarrClient | None = None,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
) -> RadarrStrategy:
    """A `RadarrStrategy` built through its real `__init__`, injecting a typed client."""

    deps = make_run_deps(config=config, cache_store=cache_store)
    services = ArrRunServices(deps, Arr.RADARR)
    return RadarrStrategy(deps, services, radarr if radarr is not None else FakeRadarrClient())


def make_sonarr_parse(**attrs: Any) -> SonarrParseCache:
    """A bare `SonarrParseCache` with `__init__` bypassed and only `attrs` set."""

    return make_bare_instance(SonarrParseCache, **attrs)


def claim_al_ids(pending: PendingImport) -> tuple[int, ...]:
    """The pending import's claiming titles, claim order."""

    return tuple(claim.al_id for claim in pending.claims)
