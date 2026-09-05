# pyright: strict
# pyright: reportPrivateUsage=false
# The factories assemble objects by their private collaborator fields (strat._episodes), which strict re-flags.
"""Builders and a bare-instance factory for the characterization tests."""

import dataclasses
import logging
from collections.abc import Iterable, Iterator
from copy import deepcopy
from datetime import datetime
from typing import Any, override

import httpx
from pydantic import BaseModel
from seadex import EntryRecord, File, Tag, TorrentRecord, Tracker

from pearlarr.anilist_client import AniListClient
from pearlarr.anilist_gateway import AniListGateway
from pearlarr.arr_categories import ArrCategoryResolver, _CategoryPair
from pearlarr.arr_http import ArrHttp
from pearlarr.cache import (
    _ENTRY_SCALAR_COLUMNS,
    UPDATED_AT_STR_FORMAT,
    AbstractCacheStore,
    CachedEntry,
    CacheRecord,
    CacheStats,
    HistoryCheckpoint,
    selection_digest_key,
)
from pearlarr.clock import Clock
from pearlarr.config import AppConfig, Arr
from pearlarr.grab_pipeline import GrabPipeline, GrabRequest
from pearlarr.import_wait import ImportProbes, ImportWaitManager, PostImportCleanup
from pearlarr.manual_import import (
    Deferral,
    GuardFacts,
    ImportProbe,
    ImportWaitMode,
    PendingImport,
    PendingKey,
)
from pearlarr.mappings import MappingResolver, MappingSources
from pearlarr.notify import Notifier
from pearlarr.output import SeverityCounts, emit_to_hub
from pearlarr.pending_records import PendingRecords
from pearlarr.planner import DownloadPlanner, PlanResult, PrivateOnlySkips
from pearlarr.radarr_client import AbstractRadarrClient
from pearlarr.reporter import RunContext, RunReporter
from pearlarr.run_services import RunDeps, RunServices
from pearlarr.seadex_filter import SeadexReleaseFilter
from pearlarr.seadex_gateway import SeaDexMiss, SeaDexSource
from pearlarr.seadex_radarr import RadarrSync
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import (
    EpisodeRecord,
    ManualImportCandidate,
    ProgressSink,
    QueueRecord,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)
from pearlarr.sonarr_client import AbstractSonarrClient
from pearlarr.sonarr_episodes import SonarrEpisodes
from pearlarr.sonarr_mapper import FileEpisodeMapper
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.torrents import AddOutcome, AddResult, TorrentService

from .fakes import FakeClock, FakeRadarrClient, FakeSonarrClient

# The display separator the cockpit/ledger/report rows join parts with. Assertions build expected
# strings from this so a separator change is one edit.
SEP = " · "

# Flat (group-local) setting name -> config group, derived from AppConfig's own field tree so it can't
# drift into a stale subset. AppConfig declares `sonarr` before `radarr`, so a name shared by both arr
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
    "import_wait_timeout": ("imports", "wait_timeout"),
    "import_ready_timeout": ("imports", "ready_timeout"),
    "import_poll_interval": ("imports", "poll_interval"),
    "import_mode": ("imports", "mode"),
    "import_default_quality": ("imports", "default_quality"),
    "import_languages_dual": ("imports", "languages_dual"),
    "import_languages_single": ("imports", "languages_single"),
    "import_pending_max_age_days": ("imports", "pending_max_age_days"),
    "wait_digest_interval": ("imports", "digest_interval"),
    "max_torrents_to_add": ("advanced", "max_torrents_to_add"),
    "sleep_time": ("advanced", "sleep_time"),
    "cache_time": ("advanced", "cache_time"),
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


# The scalar entry columns `update_cache` merges, aliased from the real store's tuple so the fake
# can't drift from `CacheStore`.
_FAKE_SCALAR_FIELDS: tuple[str, ...] = _ENTRY_SCALAR_COLUMNS


def _evict_stale[K](store: dict[K, dict[str, Any]], cutoff: datetime) -> int:
    """Drop records whose `fetched_at` is stamp-less or older than `cutoff`."""

    cutoff_str = cutoff.strftime(UPDATED_AT_STR_FORMAT)
    stale: list[K] = []
    for cache_key, record in store.items():
        stamp = record.get("fetched_at")
        if not isinstance(stamp, str) or stamp < cutoff_str:
            stale.append(cache_key)
    for cache_key in stale:
        del store[cache_key]
    return len(stale)


class FakeCacheStore(AbstractCacheStore):
    """In-memory stand-in mirroring the SQLite `CacheStore` public facade."""

    def __init__(
        self,
        *,
        sonarr_parse: dict[str, dict[str, Any]] | None = None,
        pending: dict[str, dict[PendingKey, dict[str, Any]]] | None = None,
    ) -> None:
        self._sonarr_parse: dict[str, dict[str, Any]] = dict(sonarr_parse or {})
        self._pending: dict[str, dict[PendingKey, dict[str, Any]]] = {
            arr: dict(recs) for arr, recs in (pending or {}).items()
        }
        # The entries / torrent_hashes split, keyed by (arr, al_id). An entry with an empty scalar dict
        # still "exists": the existence checks key on membership, never the dict's truthiness.
        self._entries: dict[tuple[str, int], dict[str, Any]] = {}
        self._entry_hashes: dict[tuple[str, int], list[str | None]] = {}
        self._anilist_meta: dict[int, dict[str, Any]] = {}
        self._guards: dict[str, dict[int, GuardFacts]] = {}
        self._history_checkpoints: dict[str, HistoryCheckpoint] = {}
        self._kv: dict[str, str] = {}

    # -- lifecycle --
    @override
    def save(self, *, preview: bool) -> None:
        del preview

    @override
    def close(self) -> None:
        pass

    # -- selection digest --
    @override
    def selection_stale(self, arr: Arr, digest: str) -> bool:
        stored = self._kv.get(selection_digest_key(arr))
        return stored is not None and stored != digest

    @override
    def vouch_selection(self, arr: Arr, digest: str) -> None:
        self._kv[selection_digest_key(arr)] = digest

    # -- per-entry records (entries + torrent_hashes) --
    @override
    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Partial-merge the supplied scalars. Replace the hash set if given."""

        details: dict[str, Any] = dict(cache_details) if cache_details else {}
        updated_at = details.get("updated_at")
        if isinstance(updated_at, datetime):
            details["updated_at"] = updated_at.strftime(UPDATED_AT_STR_FORMAT)

        key = (str(arr), al_id)
        entry = self._entries.setdefault(key, {})
        for column in _FAKE_SCALAR_FIELDS:
            if column in details:
                entry[column] = details[column]

        if "torrent_hashes" in details:
            hashes: list[str | None] = list(details["torrent_hashes"] or [])
            # de-dupe while keeping the single None marker the planner dedups on. The real PK leaves
            # NULLs distinct, and update_cache collapses the input to one None just like this.
            self._entry_hashes[key] = list(dict.fromkeys(hashes))

    @override
    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """True if the entry exists and its stored timestamp matches the SeaDex one."""

        sd_time_str = seadex_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT)
        entry = self._entries.get((str(arr), al_id))
        return entry is not None and entry.get("updated_at") == sd_time_str

    @override
    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None:
        """The scalar columns of the entry as a `CachedEntry`, or None."""

        entry = self._entries.get((str(arr), al_id))
        if entry is None:
            return None
        return CachedEntry(
            updated_at=entry.get("updated_at"),
            name=entry.get("name"),
            url=entry.get("url"),
            coverage=entry.get("coverage"),
            fallback_satisfied=bool(entry.get("fallback_satisfied", False)),
        )

    @override
    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]:
        """The entry's hashes, ordered None-first then ascending (mirrors ORDER BY)."""

        stored = self._entry_hashes.get((str(arr), al_id), [])
        ordered: list[str | None] = [None] if None in stored else []
        ordered.extend(sorted(h for h in stored if h is not None))
        return ordered

    # -- AniList meta (TTL-swept) --
    @override
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        yield from ((al_id, deepcopy(rec)) for al_id, rec in list(self._anilist_meta.items()))

    @override
    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        return deepcopy(self._anilist_meta.get(al_id))

    @override
    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None:
        self._anilist_meta[al_id] = deepcopy(record)

    @override
    def evict_anilist_meta(self, cutoff: datetime) -> int:
        return _evict_stale(self._anilist_meta, cutoff)

    # -- Sonarr parse cache (TTL-swept) --
    @override
    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        return deepcopy(self._sonarr_parse.get(filename))

    @override
    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None:
        self._sonarr_parse[filename] = deepcopy(record)

    @override
    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        return _evict_stale(self._sonarr_parse, cutoff)

    # -- pending imports --
    @override
    def get_pending(self, arr: Arr) -> dict[PendingKey, dict[str, Any]]:
        return {key: deepcopy(rec) for key, rec in self._pending.get(str(arr), {}).items()}

    @override
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[PendingKey, dict[str, Any]]:
        """Fresh deep-copied snapshot filtered to one series (mirrors the SQL `->> 'series_id'`)."""

        return {
            key: deepcopy(rec)
            for key, rec in self._pending.get(str(arr), {}).items()
            if rec.get("series_id") == series_id
        }

    @override
    def put_pending(self, arr: Arr, key: PendingKey, record: dict[str, Any]) -> None:
        self._pending.setdefault(str(arr), {})[key] = deepcopy(record)

    @override
    def has_pending(self, arr: Arr, key: PendingKey) -> bool:
        return key in self._pending.get(str(arr), {})

    @override
    def drop_pending(self, arr: Arr, key: PendingKey) -> None:
        self._pending.get(str(arr), {}).pop(key, None)

    @override
    def count_arr_siblings(self, arr: Arr, key: PendingKey) -> int:
        """One arr's other claims on `key`'s torrent: case-folded match, byte-exact exclusion (mirrors the SQL)."""

        target = key.infohash.casefold()
        rows = self._pending.get(str(arr), {})
        return sum(1 for other in rows if other.infohash.casefold() == target and other != key)

    @override
    def count_siblings_any_arr(self, arr: Arr, key: PendingKey) -> int:
        """Both arrs' other claims on `key`'s torrent, `arr` qualifying only the exclusion (mirrors the SQL)."""

        target = key.infohash.casefold()
        excluded = (str(arr), key)
        return sum(
            1
            for arr_key, recs in self._pending.items()
            for other in recs
            if other.infohash.casefold() == target and (arr_key, other) != excluded
        )

    # No deepcopy: GuardFacts is deeply immutable (frozen dataclass -> tuples of str/int/NamedTuple).
    @override
    def put_guards(self, arr: Arr, al_id: int, guards: GuardFacts) -> None:
        self._guards.setdefault(str(arr), {})[al_id] = guards

    @override
    def get_guards(self, arr: Arr) -> dict[int, GuardFacts]:
        """Only entries with a live pending record (mirrors the real store's join)."""

        live = {key.al_id for key in self._pending.get(str(arr), {})}
        return {al_id: g for al_id, g in self._guards.get(str(arr), {}).items() if al_id in live}

    # -- history checkpoints --
    @override
    def get_history_checkpoint(self, arr: Arr) -> HistoryCheckpoint | None:
        return self._history_checkpoints.get(str(arr))

    @override
    def put_history_checkpoint(self, arr: Arr, checkpoint: HistoryCheckpoint) -> None:
        self._history_checkpoints[str(arr)] = checkpoint

    @override
    def own_download_ids(self, arr: Arr) -> frozenset[str]:
        """Casefolded union of remembered + pending hashes (None/"" excluded)."""

        key = str(arr)
        hashes = {h.casefold() for k, hs in self._entry_hashes.items() if k[0] == key for h in hs if h}
        hashes |= {pending_key.infohash.casefold() for pending_key in self._pending.get(key, {})}
        return frozenset(hashes)

    # -- maintenance: stats, integrity --
    @override
    def stats(self) -> CacheStats:
        return CacheStats(
            entries=len(self._entries),
            torrent_hashes=sum(len(h) for h in self._entry_hashes.values()),
            anilist_meta=len(self._anilist_meta),
            sonarr_parse=len(self._sonarr_parse),
            pending_imports=sum(len(recs) for recs in self._pending.values()),
            guard_facts=sum(len(g) for g in self._guards.values()),
            size_bytes=0,
        )

    @override
    def integrity_check(self) -> str:
        return "ok"


class FakeSeaDexSource(SeaDexSource):
    """In-memory `SeaDexSource` stand-in: serves preset entries, no network."""

    def __init__(self, entries: dict[int, EntryRecord] | None = None, *, outage: bool = False) -> None:
        self._entries: dict[int, EntryRecord] = dict(entries or {})
        self._outage = outage
        self.prefetch_calls: list[list[int]] = []

    @override
    def prefetch(self, al_ids: Iterable[int], *, progress: ProgressSink | None = None) -> int:
        del progress
        ids = list(al_ids)
        self.prefetch_calls.append(ids)
        return len(ids)

    @override
    def entry(self, al_id: int) -> EntryRecord | SeaDexMiss:
        found = self._entries.get(al_id)
        if found is not None:
            return found
        return SeaDexMiss.OUTAGE if self._outage else SeaDexMiss.NO_ENTRY

    @property
    @override
    def outage(self) -> bool:
        return self._outage


def make_logger(name: str = "pearlarr-test") -> logging.Logger:
    """A quiet logger (null handler, no propagation), reset to WARNING on every call."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    return logger


def make_config(**overrides: Any) -> AppConfig:
    """An in-memory `AppConfig` carrying the decision-test defaults, `trackers` left unset (PUBLIC | PRIVATE)."""

    nested: dict[str, dict[str, Any]] = {
        "seadex": {
            "want_best": True,
            "prefer_dual_audio": True,
            "ignore_tags": [],
            "use_torrent_hash_to_filter": False,
        },
        "advanced": {"interactive": False},
    }
    for key, value in overrides.items():
        group, field = _resolve_setting(key)
        nested.setdefault(group, {})[field] = value
    return AppConfig.model_validate(nested)


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
        tags=frozenset[Tag](),
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

    counts = SeverityCounts()
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
    http: ArrHttp | None = None,
) -> ArrCategoryResolver:
    """A category resolver, fetchless by default (no transport: config values pass through, omitted stays blank).

    `http` binds a real transport for the live-fetch mode.
    """

    config = config or AppConfig()
    return ArrCategoryResolver(arr, config.for_arr(arr), http)


def make_fetched_categories(
    config: AppConfig | None = None,
    arr: Arr = Arr.SONARR,
    *,
    grab: str | None,
    post_import: str | None,
) -> ArrCategoryResolver:
    """A resolver seeded as if its one live fetch already returned `(grab, post_import)`.

    Pokes the resolver's private `_fetched` state directly, as no test builds a transport for this mode.
    """

    config = config or AppConfig()
    resolver = ArrCategoryResolver(arr, config.for_arr(arr), None)
    resolver._fetched = _CategoryPair(grab, post_import)
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
    """A real, client-less `TorrentService` (`qbit=None` -> preview no-op add)."""

    return TorrentService(qbit=None, web=web, categories=categories, tags=[], logger=logger)


def make_services(**overrides: Any) -> RunServices:
    """A bare `RunServices` carrying only the attributes its methods read."""

    logger = make_logger()

    config = _split_config(overrides)
    defaults: dict[str, Any] = {
        "logger": logger,
        "_config": config,
        # The real __init__ always sets the authoritative arr and a RunContext. Override _ctx=... for a run state.
        "arr": Arr.SONARR,
        "_ctx": RunContext(arr=Arr.SONARR),
        # The real __init__ mints these too. Without them the dirty-aware / selection-aware skip predicates fail.
        "_dirty_al_ids": set[int](),
        "_selection_stale": False,
    }
    defaults.update(overrides)
    return make_bare_instance(RunServices, **defaults)


def make_run_deps(
    *,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    seadex: SeaDexSource | None = None,
    logger: logging.Logger | None = None,
    clock: Clock | None = None,
) -> RunDeps:
    """A real `RunDeps` over typed fakes, with a Sonarr url and api_key set and `qbit` None (preview)."""

    config = config or make_config(url="http://sonarr", api_key="key")
    cache_store = cache_store or FakeCacheStore()
    logger = logger or make_logger()
    # One shared client backs both deps.http and deps.web. conftest's close_leaked_handles closes it at teardown.
    http = httpx.Client()
    # No transport: the config passthrough (no download-client fetch under test). Shared with
    # torrents below, matching production's single resolver instance.
    categories = make_categories(config)
    return RunDeps(
        config=config,
        arr_config=config.for_arr(Arr.SONARR),
        # None: the strategies' require_connection fallback binds lazily, so the keys-missing
        # construction seams behave exactly as production's.
        arr_http=None,
        categories=categories,
        clock=clock or FakeClock(),
        web=http,
        http=http,
        qbit=None,
        # A real resolver over empty in-memory mappings (no network). It carries a real (empty)
        # `anibridge` the strategy reads at construction.
        mappings=MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=http,
            sources=MappingSources(anime={}, anidb=False, anibridge=False),
        ),
        logger=logger,
        seadex=seadex or FakeSeaDexSource(),
        cache_store=cache_store,
        anilist=AniListGateway(cache_store=cache_store, logger=logger, client=AniListClient(client=http)),
        torrents=_real_torrents(logger, http, categories),
        notifier=Notifier(discord_url=None, webhook_url=None, web=http),
        planner=make_planner(),
        reporter=_real_reporter(logger, cache_store, http),
    )


def make_release_filter(**overrides: Any) -> SeadexReleaseFilter:
    """A `SeadexReleaseFilter` over an assembled `RunDeps`, with `ctx`, `cache_store`, and `planner` overridable."""

    config = _split_config(overrides)
    ctx = overrides.pop("ctx", None) or RunContext(arr=Arr.SONARR)
    deps = make_run_deps(config=config, cache_store=overrides.pop("cache_store", None))
    if "planner" in overrides:
        deps = dataclasses.replace(deps, planner=overrides.pop("planner"))
    if overrides:
        msg = f"unknown make_release_filter overrides: {sorted(overrides)}"
        raise TypeError(msg)
    return SeadexReleaseFilter(deps=deps, ctx=ctx)


# A truthy stand-in for a logged-in qBittorrent client, so is_preview() is False.
CLIENT_SENTINEL = object()


class FakeTorrents:
    """Mimics `TorrentService.add`: a per-hash scripted `(outcome, name)` or full `AddResult`."""

    def __init__(
        self,
        by_hash: dict[str | None, tuple[AddOutcome, str | None] | AddResult],
        *,
        raises: dict[str | None, Exception] | None = None,
    ) -> None:
        self._by_hash = by_hash
        self._raises = raises or {}
        self.calls: list[str | None] = []

    def add(
        self,
        *,
        item: SeadexUrlItem,
        preview: bool,
    ) -> AddResult:
        """Return the scripted `AddResult` for the url item's infohash, or raise its scripted error."""

        del preview
        infohash = item.infohash
        self.calls.append(infohash)
        if infohash in self._raises:
            raise self._raises[infohash]
        scripted = self._by_hash[infohash]
        return scripted if isinstance(scripted, AddResult) else AddResult(*scripted)


def one_release_dict(*, srg: str, infohash: str, url: str = "https://nyaa.si/view/1") -> SeadexDict:
    """A one-release `SeadexDict` flagged for download, its tracker pinned to `NYAA`."""

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return {srg: rg_group({url: item})}


def grab_request(**overrides: Any) -> GrabRequest:
    """A minimal `GrabRequest` for driving the add path directly (an empty `seadex_dict`, no seeds)."""

    defaults: dict[str, Any] = {
        "al_id": 1,
        "item_title": "Show",
        "anilist_title": "Show",
        "entry": make_entry_record(url="https://releases.moe/1"),
        "seadex_dict": {},
        "torrent_hashes": [],
        "cache_details": {},
        "replaced_groups": (),
    }
    defaults.update(overrides)
    return GrabRequest(**defaults)


def make_grab_pipeline(**overrides: Any) -> GrabPipeline:
    """A bare `GrabPipeline` carrying only what its methods read, `_ctx` a non-preview blocking run."""

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
        "_ctx": RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING),
    }
    defaults.update(overrides)
    if "_records" not in defaults:
        # The record seam binds the FINAL store + ctx, exactly as the real ctor + begin_run do.
        records = PendingRecords(defaults["cache_store"])
        records.begin_run(defaults["_ctx"])
        defaults["_records"] = records
    return make_bare_instance(GrabPipeline, **defaults)


def make_import_wait_manager(**overrides: Any) -> ImportWaitManager:
    """A bare `ImportWaitManager` over real sub-objects, config keys routed through `AppConfig`.

    Constructs the records/probes/cleanup seams on the SAME store and ends with
    `begin_run`, so every sub-object shares the one ctx (and strategy) a run would.
    """

    config = overrides.pop("config", None) or _split_config(overrides)
    logger = overrides.pop("logger", None) or make_logger()
    cache_store = overrides.pop("cache_store", None) or FakeCacheStore()
    qbit = overrides.pop("qbit", None)
    clock = overrides.pop("clock", None) or FakeClock()
    # A fetchless resolver (`RunDeps.categories`): the config category passes through, the
    # arr-client fallback being an `ArrCategoryResolver` concern.
    categories = overrides.pop("categories", None) or make_categories(config)
    # The production placeholder before a run binds one. Tests driving the import hook pass their own.
    strategy = overrides.pop("strategy", None)
    ctx = overrides.pop("ctx", None) or RunContext(arr=Arr.SONARR)
    reporter = overrides.pop("reporter", None) or _real_reporter(logger, cache_store, httpx.Client())
    if overrides:
        msg = f"unknown make_import_wait_manager overrides: {sorted(overrides)}"
        raise TypeError(msg)

    records = PendingRecords(cache_store)
    probes = make_bare_instance(ImportProbes, _qbit=qbit, _logger=logger, strategy=None)
    cleanup = make_bare_instance(
        PostImportCleanup,
        _imports=config.imports,
        _categories=categories,
        _qbit=qbit,
        _clock=clock,
        _logger=logger,
        _records=records,
        _probes=probes,
    )
    mgr = make_bare_instance(
        ImportWaitManager,
        imports=config.imports,
        clock=clock,
        logger=logger,
        _reporter=reporter,
        _records=records,
        probes=probes,
        _cleanup=cleanup,
    )
    mgr.begin_run(ctx, strategy)
    return mgr


def make_planner(**overrides: Any) -> DownloadPlanner:
    """A `DownloadPlanner` with test-friendly defaults (arr `SONARR`, both flags off)."""

    logger = make_logger()

    defaults: dict[str, Any] = {
        "arr": Arr.SONARR,
        "interactive": False,
        "use_torrent_hash_to_filter": False,
        "logger": logger,
    }
    defaults.update(overrides)
    return DownloadPlanner(**defaults)


def url_item(
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
    episodes: list[EpisodeRecord] | None = None,
) -> SeadexUrlItem:
    """One SeaDex URL record, matching `get_seadex_dict`'s `url_item` shape."""

    return SeadexUrlItem(
        url=url,
        files=files or [],
        size=size or [],
        tracker=tracker,
        is_public=is_public,
        is_dual_audio=is_dual_audio,
        infohash=infohash,
        download=download,
        is_fallback=is_fallback,
        upgrade=upgrade,
        episodes=episodes or [],
    )


def plan_result(torrent_hashes: list[str | None], seadex_dict: SeadexDict) -> PlanResult:
    """A `PlanResult` carrying just the hashes and the dict, with empty `PrivateOnlySkips`."""

    return PlanResult(seadex_dict=seadex_dict, torrent_hashes=torrent_hashes, skips=PrivateOnlySkips())


def rg_group(
    urls: dict[str, SeadexUrlItem],
    *,
    tags: frozenset[Tag] | None = None,
    all_episodes: list[EpisodeRecord] | None = None,
) -> SeadexReleaseGroupItem:
    """`all_episodes` is tri-state: `None` skips parsing, `[]` is unparsed, populated is the coverage frozenset."""

    return SeadexReleaseGroupItem(
        urls=urls,
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
) -> SonarrEpisode:
    """`episode_file_id=0` omits the `episodeFile` block, as in Sonarr's record for a missing episode."""

    raw: dict[str, Any] = {
        "id": ep_id,
        "seasonNumber": season,
        "episodeNumber": episode,
        "episodeFileId": episode_file_id,
    }
    if episode_file_id:
        raw["episodeFile"] = {"size": size, "releaseGroup": release_group}
    return SonarrEpisode.model_validate(raw)


# The `al_id` every `pending_import` record carries unless overridden, exported so tests can spell
# a builder record's composite key without magic ints.
PENDING_AL_ID = 1


def pending_import(**overrides: Any) -> PendingImport:
    """A `PendingImport` wiring one mapped file to one episode id, with a matching flat fallback."""

    defaults: dict[str, Any] = {
        "infohash": "abc123",
        "series_id": 7,
        "al_id": PENDING_AL_ID,
        "file_episode_map": {"Show - 01 [1080p].mkv": [101]},
        "episode_ids": [101],
        "release_group": "SubGroup",
        "is_dual_audio": False,
        "seadex_files": ["Show - 01 [1080p].mkv"],
        "title": "Show",
        "added_at": "2026-06-24 00:00:00",
    }
    defaults.update(overrides)
    return PendingImport(**defaults)


def import_probe(
    *,
    files_present: bool = True,
    command_issued: bool = False,
    imported_count: int = 0,
    target_count: int = 0,
    deferral: Deferral = Deferral.NONE,
    placements: dict[str, list[int]] | None = None,
) -> ImportProbe:
    """An `ImportProbe` defaulting to the verified-import outcome (`files_present`)."""

    return ImportProbe(
        files_present=files_present,
        command_issued=command_issued,
        imported_count=imported_count,
        target_count=target_count,
        deferral=deferral,
        placements=placements or {},
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

    defaults: dict[str, Any] = {"_ep_list_cache": {}, "_series_fp": ""}
    defaults.update(attrs)
    return make_bare_instance(SonarrEpisodes, **defaults)


def make_sonarr_sync(
    *,
    sonarr: AbstractSonarrClient | None = None,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    ep_list_cache: dict[int, list[SonarrEpisode]] | None = None,
    clock: Clock | None = None,
) -> SonarrSync:
    """A `SonarrSync` built through its real `__init__`, injecting a typed client and seeding `ep_list_cache`."""

    deps = make_run_deps(config=config, cache_store=cache_store, clock=clock)
    services = RunServices(deps, Arr.SONARR)
    strat = SonarrSync(
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
) -> RadarrSync:
    """A `RadarrSync` built through its real `__init__`, injecting a typed client."""

    deps = make_run_deps(config=config, cache_store=cache_store)
    services = RunServices(deps, Arr.RADARR)
    return RadarrSync(deps, services, radarr if radarr is not None else FakeRadarrClient())


def make_sonarr_mapper(**attrs: Any) -> FileEpisodeMapper:
    """A bare `FileEpisodeMapper` with `__init__` bypassed, its parse-info cache defaulted empty."""

    defaults: dict[str, Any] = {"_parse_info_cache": {}}
    defaults.update(attrs)
    return make_bare_instance(FileEpisodeMapper, **defaults)


def make_sonarr_parse(**attrs: Any) -> SonarrParseCache:
    """A bare `SonarrParseCache` with `__init__` bypassed and only `attrs` set."""

    return make_bare_instance(SonarrParseCache, **attrs)
