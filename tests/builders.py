# pyright: strict
# pyright: reportPrivateUsage=false
# The factories assemble objects under construction by their private collaborator
# fields (strat._episodes/_parse/...), which strict re-flags; the repo disables
# reportPrivateUsage for tests.
"""Builders and a bare-instance factory for the characterization tests.

These tests pin the *current* behaviour of the run machinery. The planner tests
build its inputs (typed episode records, flat release dicts) via the helpers
here, and ``make_services`` builds a ``RunServices`` without running its heavy
``__init__`` (network downloads, qBittorrent login, disk I/O), assigning only
the attributes the methods under test actually read.
"""

import dataclasses
import logging
from collections.abc import Iterable, Iterator
from copy import deepcopy
from datetime import datetime
from typing import Any, override

import httpx
from seadex import EntryRecord, File, Tag, TorrentRecord, Tracker

from seadexarr.modules.anilist_gateway import AniListGateway
from seadexarr.modules.cache import (
    _ENTRY_SCALAR_COLUMNS,
    UPDATED_AT_STR_FORMAT,
    AbstractCacheStore,
    CachedEntry,
    CacheRecord,
    CacheStats,
    HistoryCheckpoint,
)
from seadexarr.modules.config import AppConfig, Arr
from seadexarr.modules.grab_pipeline import GrabPipeline
from seadexarr.modules.import_wait import ImportWaitManager
from seadexarr.modules.log import LogCounter, LogFormatter
from seadexarr.modules.manual_import import ImportProbe, ImportReadiness, ImportWaitMode, PendingImport
from seadexarr.modules.mappings import MappingResolver, MappingSources
from seadexarr.modules.notify import Notifier
from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.reporter import RunContext, RunReporter
from seadexarr.modules.run_services import RunDeps, RunServices
from seadexarr.modules.seadex_filter import SeadexReleaseFilter
from seadexarr.modules.seadex_gateway import SeaDexMiss, SeaDexSource
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    EpisodeRecord,
    ManualImportCandidate,
    ProgressSink,
    QualityModel,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)
from seadexarr.modules.sonarr_client import AbstractSonarrClient
from seadexarr.modules.sonarr_episodes import SonarrEpisodes
from seadexarr.modules.sonarr_mapper import FileEpisodeMapper
from seadexarr.modules.sonarr_parse import SonarrParseCache
from seadexarr.modules.torrents import AddOutcome, AddResult, TorrentService

from .fakes import FakeSonarrClient

# The " · " display separator the cockpit/ledger/report rows join parts with.
# Assertions build expected strings from this so a separator change is one edit.
SEP = " · "

# Map each flat (group-local) setting name to its config group, derived straight from
# AppConfig's own field tree so it can't drift into a stale subset: adding a 9th
# settings group to AppConfig wires it in here for free. AppConfig declares ``sonarr``
# before ``radarr``, so a name shared across the two arr groups (the ArrSettings keys
# url/api_key/ignore_unmonitored/torrent_category) resolves to ``sonarr`` - the arr
# make_config/make_services default to - via the first-wins ``setdefault`` below.
_FIELD_GROUP: dict[str, str] = {}
for _group, _group_field in AppConfig.model_fields.items():
    # ``annotation`` is the group's submodel class (typed Optional on FieldInfo, but
    # every AppConfig group field is a concrete ``_ConfigBase`` subclass); read its
    # own field names off it.
    _submodel: Any = _group_field.annotation
    for _field in _submodel.model_fields:
        _FIELD_GROUP.setdefault(_field, _group)

# Pre-nesting flat names -> (group, field). The historical builder interface used
# the old flat config keys (``import_*``, ``{arr}_*``, etc.); map them here so the
# existing call sites keep passing flat kwargs without each one being rewritten.
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
    # The bare url/api_key flat names resolve to sonarr (first-wins); these reach
    # the radarr connection keys for the Radarr-run builders/tests.
    "radarr_url": ("radarr", "url"),
    "radarr_api_key": ("radarr", "api_key"),
}


def _resolve_setting(key: str) -> tuple[str, str]:
    """Map a flat override key to its ``(group, field)`` in the nested config."""

    if key in _FLAT_ALIASES:
        return _FLAT_ALIASES[key]
    return _FIELD_GROUP.get(key, "seadex"), key


# The override keys make_services routes into self._config (rather than onto the
# bare instance as a direct attribute/collaborator).
_CONFIG_SETTING_NAMES = frozenset(_FIELD_GROUP) | frozenset(_FLAT_ALIASES)


def _split_config(overrides: dict[str, Any]) -> AppConfig:
    """Pop the config-routed keys out of ``overrides`` (IN PLACE) and build the config.

    Mutates ``overrides``: the builders' later ``defaults.update(overrides)`` relies on
    the config keys having been removed - else e.g. ``private_releases`` would be set as
    a bare engine attribute the code never reads instead of routing to ``self._config``.
    """

    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}
    return make_config(**config_overrides)


def make_bare_instance[T](cls: type[T], **attrs: Any) -> T:
    """An instance with ``__init__`` bypassed and only the given attrs set.

    ``object.__new__`` skips the real, heavy ``__init__`` (network downloads,
    qBittorrent login, disk I/O); the tests assign just the attributes the
    methods under test read. Shared by ``make_services`` here and the
    strategy-seam tests so the bypass idiom lives in one place.
    """

    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


# The scalar entry columns ``update_cache`` merges: the real store's own tuple, so
# the fake can't drift from ``CacheStore``.
_FAKE_SCALAR_FIELDS: tuple[str, ...] = _ENTRY_SCALAR_COLUMNS


def _evict_stale[K](store: dict[K, dict[str, Any]], cutoff: datetime) -> int:
    """Drop records whose ``fetched_at`` is stamp-less or older than ``cutoff``.

    Mirrors the real ``DELETE ... WHERE fetched_at < ? OR fetched_at IS NULL``: a
    missing/None (or otherwise non-string) stamp is unreadable and so swept too.
    """

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
    """In-memory stand-in mirroring the SQLite ``CacheStore`` public facade.

    Backs every facade block - the per-entry ``entries`` scalars plus their
    ``torrent_hashes`` child set, the ``anilist_meta`` and ``sonarr_parse`` JSONB
    caches, and ``pending_imports`` - with plain dicts, so a driven path that
    reaches ANY facade method gets the real store's behaviour instead of an
    ``AttributeError`` or a silent no-op. Semantics are matched, not just the
    names: ``update_cache`` partial-merges the supplied scalars and (when given)
    REPLACES the whole hash set while keeping a single ``None`` marker;
    ``check_al_id_in_cache`` compares the strftime'd ``updated_at`` strings; and the
    ``evict_*`` sweeps drop stamp-less / aged-out records like the real SQL DELETE.

    Every JSONB record block (``pending`` / ``anilist_meta`` / ``sonarr_parse``)
    deep-copies on BOTH ends - stored on ``put_*`` and returned on ``get_*`` /
    ``iter_*`` - so a caller mutating a record before or after the call cannot reach
    the store, mirroring the real store's ``json.dumps`` / ``json.loads`` round-trip.
    ``save`` / ``close`` are no-ops; ``stats`` / ``integrity_check`` report a plausible
    health snapshot. Arr keys use ``str(arr)`` to mirror production's ``_arr_key``.
    """

    def __init__(
        self,
        *,
        sonarr_parse: dict[str, dict[str, Any]] | None = None,
        pending: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._sonarr_parse: dict[str, dict[str, Any]] = dict(sonarr_parse or {})
        self._pending: dict[str, dict[str, dict[str, Any]]] = {arr: dict(recs) for arr, recs in (pending or {}).items()}
        # Per-entry records: the scalar columns keyed by (arr, al_id), and the
        # entry's torrent-hash set kept separately (the entries / torrent_hashes
        # split). An entry present with an empty scalar dict still "exists" - the
        # existence checks key on membership, never the dict's truthiness.
        self._entries: dict[tuple[str, int], dict[str, Any]] = {}
        self._entry_hashes: dict[tuple[str, int], list[str | None]] = {}
        self._anilist_meta: dict[int, dict[str, Any]] = {}
        self._history_checkpoints: dict[str, HistoryCheckpoint] = {}

    # -- lifecycle --
    @override
    def save(self, *, preview: bool) -> None:
        del preview

    @override
    def close(self) -> None:
        pass

    # -- per-entry records (entries + torrent_hashes) --
    @override
    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Partial-merge the supplied scalars; replace the hash set if given."""

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
            # dict.fromkeys de-dupes while keeping the single None marker the
            # planner dedups on (the real PK leaves NULLs distinct; update_cache
            # collapses the input to one None just like this).
            self._entry_hashes[key] = list(dict.fromkeys(hashes))

    @override
    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """True iff the entry exists and its stored timestamp matches the SeaDex one."""

        sd_time_str = seadex_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT)
        entry = self._entries.get((str(arr), al_id))
        return entry is not None and entry.get("updated_at") == sd_time_str

    @override
    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None:
        """The scalar columns of the entry as a ``CachedEntry``, or None."""

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
    def get_pending(self, arr: Arr) -> dict[str, dict[str, Any]]:
        return {ih: deepcopy(rec) for ih, rec in self._pending.get(str(arr), {}).items()}

    @override
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]:
        """Fresh deep-copied snapshot filtered to one series (mirrors the SQL ``->> 'series_id'``)."""

        return {
            ih: deepcopy(rec)
            for ih, rec in self._pending.get(str(arr), {}).items()
            if rec.get("series_id") == series_id
        }

    @override
    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None:
        self._pending.setdefault(str(arr), {})[infohash] = deepcopy(record)

    @override
    def drop_pending(self, arr: Arr, infohash: str) -> None:
        self._pending.get(str(arr), {}).pop(infohash, None)

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
        hashes |= {ih.casefold() for ih in self._pending.get(key, {})}
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
            size_bytes=0,
        )

    @override
    def integrity_check(self) -> str:
        return "ok"


class FakeSeaDexSource(SeaDexSource):
    """In-memory ``SeaDexSource`` stand-in: serves preset entries, no network.

    Retires ``make_run_deps``'s ``make_bare_instance(SeaDexGateway)`` landmine - a
    zero-attribute bare instance laundered to ``Any``, every access an
    ``AttributeError`` waiting to happen. Backed by a plain ``{al_id: EntryRecord}``
    map: ``entry`` serves from it (a miss is NO_ENTRY, or OUTAGE when constructed
    with ``outage=True``, mirroring the real gateway's short-circuit); ``prefetch``
    is a no-op that records the ids and reports their count (mirroring the real
    "how many needed fetching" return).
    """

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


def make_logger(name: str = "seadexarr-test") -> logging.Logger:
    """A quiet logger for the characterization tests.

    Attaches a NullHandler, disables propagation, and resets the level to
    WARNING on every call so the hot-path debug f-strings aren't formatted and a
    test that bumps the level can't leak into the next. A ``LogCounter`` filter
    rides along (as on a ``setup_logger`` logger), so the run-summary /
    run-loop counter readers (``log_counter``) work against test loggers too.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    if not any(isinstance(f, LogCounter) for f in logger.filters):
        logger.addFilter(LogCounter())
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    return logger


def make_config(**overrides: Any) -> AppConfig:
    """An in-memory ``AppConfig`` carrying the decision-test defaults.

    The config flags are read through ``self._config`` (the single source of truth),
    so a bare instance needs a real ``AppConfig``. These defaults mirror the historical
    ``make_arr`` flags and leave ``trackers`` unset so it defaults to PUBLIC | PRIVATE.
    Each flat override is routed to its config group (``_FIELD_GROUP``) and the
    nested mapping is validated through the models, so the before-validators run exactly
    as on a real load.
    """

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
) -> EntryRecord:
    """A real ``seadex.EntryRecord`` with the 13 required fields defaulted.

    The library type is a frozen ``msgspec.Struct``, so it can't be duck-typed
    under strict; this builds the real value with sane defaults and exposes the
    handful of fields tests vary (``url``/``is_incomplete``/``updated_at``/
    ``torrents``). Replaces the old per-file ``_FakeEntry`` stand-ins.
    """

    stamp = updated_at if updated_at is not None else datetime(2026, 1, 1)
    return EntryRecord(
        anilist_id=anilist_id,
        collection_id="col",
        collection_name="col-name",
        comparisons=(),
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
    is_dual_audio: bool = False,
    is_best: bool = True,
    size: int = 1000,
) -> TorrentRecord:
    """A real ``seadex.TorrentRecord`` (frozen msgspec) with sane release defaults.

    ``file_names`` are wrapped into ``seadex.File`` entries (each ``file_size`` bytes)
    so a caller seeds the on-disk file list the Sonarr matching parses, without
    importing the library leaf types itself.
    """

    stamp = datetime(2026, 1, 1)
    return TorrentRecord(
        collection_id="c",
        collection_name="cn",
        created_at=stamp,
        is_dual_audio=is_dual_audio,
        files=tuple(File(name=name, size=file_size) for name in file_names),
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
    log_fmt: LogFormatter,
    cache_store: AbstractCacheStore,
    web: httpx.Client,
) -> RunReporter:
    """A real ``RunReporter`` over the given cache store (composite-with-faked-leaf).

    The reporter is a presentation collaborator with a large surface; rather than
    fake it, the factories build the real one with a faked cache store + a real
    (cache-backed) AniList gateway - so a driven path logs through the real code.
    """

    return RunReporter(
        log_fmt=log_fmt,
        cache_store=cache_store,
        anilist=AniListGateway(cache_store=cache_store, logger=logger, web=web),
    )


def _real_torrents(logger: logging.Logger, web: httpx.Client) -> TorrentService:
    """A real, client-less ``TorrentService`` (``qbit=None`` -> preview no-op add)."""

    return TorrentService(qbit=None, web=web, category="", tags=[], logger=logger)


def make_services(**overrides: Any) -> RunServices:
    """Build a bare ``RunServices`` with only the attributes the methods read.

    Bypasses ``__init__`` via ``object.__new__`` and assigns sane defaults for
    the collaborators the per-id decision methods consult; the config flags live
    on an in-memory ``AppConfig`` (``self._config``). Pass keyword overrides to
    vary a single config flag (e.g. ``make_services(private_releases="warn")``)
    or another attribute.
    """

    logger = make_logger()

    # Config-backed flags are read via self._config; route any passed as
    # overrides through an in-memory AppConfig (popped from overrides in place),
    # leaving the rest as direct attributes/collaborators.
    config = _split_config(overrides)
    defaults: dict[str, Any] = {
        "logger": logger,
        "log_fmt": LogFormatter(logger),
        "_config": config,
        # The real __init__ always sets the authoritative arr + a RunContext;
        # faithful defaults so methods that read self._ctx.arr (the run-arr
        # methods) work without each test wiring one. Override with _ctx=... for
        # a specific run state.
        "arr": Arr.SONARR,
        "_ctx": RunContext(arr=Arr.SONARR),
        # The real __init__ always mints this; bare instances need it too or the
        # dirty-aware skip predicates fail at runtime.
        "_dirty_al_ids": set[int](),
    }
    defaults.update(overrides)
    return make_bare_instance(RunServices, **defaults)


def make_run_deps(
    *,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    seadex: SeaDexSource | None = None,
    logger: logging.Logger | None = None,
) -> RunDeps:
    """A real ``RunDeps`` (typed fakes) to drive the REAL ``RunServices`` /
    ``RunLoop`` / ``SonarrSync`` ``__init__`` + ``begin_run`` rebind - the
    construction seam ``make_bare_instance`` bypasses.

    Every field is passed to ``RunDeps`` by explicit keyword (no ``**dict[str, Any]``
    launder), so each is type-checked against the dataclass field at this seam - a
    wrong-typed fake (``cache_store=object()``, a non-``SeaDexSource`` seadex) is a
    pyright error here, not a silent ``Any``. The config carries a Sonarr
    url/api_key so ``SonarrSync``'s ``require_connection`` passes; ``qbit`` is
    ``None`` (preview, no auth); ``cache_store`` defaults to the in-memory
    ``FakeCacheStore`` so the staged-write sharing can be asserted by identity;
    ``seadex`` to a network-free ``FakeSeaDexSource``.
    """

    config = config or make_config(url="http://sonarr", api_key="key")
    cache_store = cache_store or FakeCacheStore()
    logger = logger or make_logger()
    log_fmt = LogFormatter(logger)
    # The one deliberately-leaked httpx client backs BOTH deps.http and
    # deps.web (never used for real traffic here; httpx clients don't warn on GC).
    http = httpx.Client()
    return RunDeps(
        config=config,
        arr_config=config.for_arr(Arr.SONARR),
        web=http,
        http=http,
        qbit=None,
        # A real resolver over empty in-memory mappings (no network) - it carries a
        # real (empty) ``anibridge`` the strategy reads at construction.
        mappings=MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            sources=MappingSources(anime={}, anidb=False, anibridge=False),
        ),
        logger=logger,
        # A typed, network-free SeaDex stand-in (the wiring tests never look one up);
        # retires the old make_bare_instance(SeaDexGateway) Any-launder.
        seadex=seadex or FakeSeaDexSource(),
        cache_store=cache_store,
        anilist=AniListGateway(cache_store=cache_store, logger=logger, web=http),
        torrents=_real_torrents(logger, http),
        notifier=Notifier(discord_url=None, webhook_url=None, web=http, logger=logger),
        planner=make_planner(),
        log_fmt=log_fmt,
        reporter=_real_reporter(logger, log_fmt, cache_store, http),
    )


def make_release_filter(**overrides: Any) -> SeadexReleaseFilter:
    """Build a ``SeadexReleaseFilter`` over an assembled ``RunDeps``.

    Config-backed flags (e.g. ``want_best``, ``private_releases``) route through
    an in-memory ``AppConfig``; ``cache_store``/``planner`` override the deps
    fields the real ctor unpacks and ``ctx`` the run context. Mirrors
    ``make_services``'s override routing so the ``build`` characterization tests
    read the same as the old ``get_seadex_dict`` ones.
    """

    config = _split_config(overrides)
    ctx = overrides.pop("ctx", None) or RunContext(arr=Arr.SONARR)
    deps = make_run_deps(config=config, cache_store=overrides.pop("cache_store", None))
    if "planner" in overrides:
        deps = dataclasses.replace(deps, planner=overrides.pop("planner"))
    if overrides:
        # Preserve the old **kwargs ctor's fail-loud contract for unknown keys.
        msg = f"unknown make_release_filter overrides: {sorted(overrides)}"
        raise TypeError(msg)
    return SeadexReleaseFilter(deps=deps, ctx=ctx)


# A truthy stand-in for a logged-in qBittorrent client, so is_preview() is
# False without a real login (the actual add is faked by FakeTorrents).
CLIENT_SENTINEL = object()


class FakeTorrents:
    """Mimics ``TorrentService.add``: a per-hash scripted ``(outcome, name)``.

    Keyed by infohash (not call order) so a multi-release add can return a
    different outcome per release regardless of dict iteration order. A hash
    scripted in ``raises`` raises its exception instead (a tracker/client
    failure the pipeline's containment must absorb).
    """

    def __init__(
        self,
        by_hash: dict[str | None, tuple[AddOutcome, str | None]],
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
        del preview
        infohash = item.infohash
        self.calls.append(infohash)
        if infohash in self._raises:
            raise self._raises[infohash]
        return AddResult(*self._by_hash[infohash])


def one_release_dict(*, srg: str, infohash: str, url: str = "https://nyaa.si/view/1") -> SeadexDict:
    """A one-release ``SeadexDict`` flagged for download on a public tracker.

    The builder defaults the tracker to ``OTHER`` (not in the selected set), so
    pin it to ``NYAA`` to clear ``_add_one_url``'s tracker filter.
    """

    item = url_item(url=url, infohash=infohash, download=True)
    item.tracker = Tracker.NYAA
    return {srg: rg_group({url: item})}


def make_grab_pipeline(**overrides: Any) -> GrabPipeline:
    """Build a bare ``GrabPipeline`` with only what its methods read.

    Mirrors ``make_release_filter``: config-backed flags route through an
    in-memory ``AppConfig``; the rest pass straight to the bare instance. The
    ``_ctx`` defaults to a non-preview blocking run (so the pending-import
    registration is reachable); pass ``qbit=CLIENT_SENTINEL`` (the default) for a
    non-preview run or ``qbit=None`` to exercise the preview short-circuit.
    """

    config = _split_config(overrides)
    logger = make_logger()
    log_fmt = LogFormatter(logger)
    cache_store = overrides.pop("cache_store", None) or FakeCacheStore()
    web = httpx.Client()
    defaults: dict[str, Any] = {
        "_config": config,
        "_planner": make_planner(),
        "cache_store": cache_store,
        "_torrents": _real_torrents(logger, web),
        "_anilist": AniListGateway(cache_store=cache_store, logger=logger, web=web),
        # No discord/webhook url -> a disabled, best-effort no-op notifier.
        "_notifier": Notifier(discord_url=None, webhook_url=None, web=web, logger=logger),
        "_reporter": _real_reporter(logger, log_fmt, cache_store, web),
        "log_fmt": log_fmt,
        "qbit": CLIENT_SENTINEL,
        "_ctx": RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING),
        # __init__-seeded per-title state the bare instance must also carry.
        "_grab_failed_groups": [],
    }
    defaults.update(overrides)
    return make_bare_instance(GrabPipeline, **defaults)


def make_import_wait_manager(**overrides: Any) -> ImportWaitManager:
    """Build a bare ``ImportWaitManager`` with only what its methods read.

    Config-backed flags (the import timeouts / poll interval) route through an
    in-memory ``AppConfig``; the rest pass straight to the bare instance. The
    ``_ctx`` defaults to a fresh Sonarr run context and ``_active_strategy`` /
    ``_reporter`` / ``cache_store`` to mocks/fakes, so a test sets just the
    qbit + strategy + store it exercises.
    """

    config = _split_config(overrides)
    logger = make_logger()
    cache_store = overrides.pop("cache_store", None) or FakeCacheStore()
    defaults: dict[str, Any] = {
        "_config": config,
        "cache_store": cache_store,
        "_reporter": _real_reporter(logger, LogFormatter(logger), cache_store, httpx.Client()),
        "logger": logger,
        "qbit": None,
        "_ctx": RunContext(arr=Arr.SONARR),
        # The production placeholder before a run binds one; tests that drive the
        # import hook pass their own strategy.
        "_active_strategy": None,
    }
    defaults.update(overrides)
    return make_bare_instance(ImportWaitManager, **defaults)


def make_planner(**overrides: Any) -> DownloadPlanner:
    """Build a ``DownloadPlanner`` with test-friendly defaults.

    The planner reads its bound arr (default ``SONARR``; override with
    ``arr=Arr.RADARR``) and two config flags plus a logger; pass keyword
    overrides to vary a single flag (e.g. ``make_planner(interactive=True)``).
    The logger defaults to WARNING so the hot-path debug f-strings aren't
    formatted, mirroring ``make_services``.
    """

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
    infohash: str | None = "hash1",
    download: bool = False,
    is_fallback: bool = False,
    size_mismatch: bool = False,
    episodes: list[EpisodeRecord] | None = None,
) -> SeadexUrlItem:
    """One SeaDex URL record, matching ``get_seadex_dict``'s ``url_item`` shape."""

    return SeadexUrlItem(
        url=url,
        files=files or [],
        size=size or [],
        tracker=tracker,
        is_public=is_public,
        infohash=infohash,
        download=download,
        is_fallback=is_fallback,
        size_mismatch=size_mismatch,
        episodes=episodes or [],
    )


def rg_group(
    urls: dict[str, SeadexUrlItem],
    *,
    tags: frozenset[Tag] | None = None,
    all_episodes: list[EpisodeRecord] | None = None,
) -> SeadexReleaseGroupItem:
    """One SeaDex release-group record keyed by url.

    ``all_episodes`` defaults to ``None`` so the three branches of
    ``get_same_files_groups`` (``None`` -> no-parsing, ``[]`` -> unparsed,
    populated -> coverage frozenset) can each be reached.
    """

    return SeadexReleaseGroupItem(
        urls=urls,
        tags=tags or frozenset(),
        all_episodes=all_episodes,
    )


def sonarr_ep(
    season: int,
    episode: int,
    *,
    size: int | None = None,
    release_group: str | None = None,
    episode_file_id: int = 1,
) -> SonarrEpisode:
    """One ``SonarrEpisode``, parsed from the raw fields the engine reads."""

    return SonarrEpisode.from_api(
        {
            "seasonNumber": season,
            "episodeNumber": episode,
            "episodeFileId": episode_file_id,
            "episodeFile": {"size": size, "releaseGroup": release_group},
        },
    )


def pending_import(**overrides: Any) -> PendingImport:
    """A ``PendingImport`` carrying sane manual-import defaults.

    Defaults wire one mapped file to a single episode id with a matching flat
    fallback, dual-audio off, and a single season; pass keyword overrides to vary
    any field (e.g. ``pending_import(is_dual_audio=True)``).
    """

    defaults: dict[str, Any] = {
        "infohash": "abc123",
        "series_id": 7,
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
    readiness: ImportReadiness = ImportReadiness.IMPORTED,
    *,
    files_present: bool = True,
    command_issued: bool = False,
    imported_count: int = 0,
    target_count: int = 0,
) -> ImportProbe:
    """An :class:`ImportProbe` with the common "files verified imported" defaults.

    The default is the verified-import outcome (``IMPORTED`` + ``files_present``);
    pass ``readiness=RETRY, files_present=False, command_issued=True`` for the
    "command accepted, copy in flight" case, or ``files_present=False`` for a not-
    yet-ready poll. ``imported_count``/``target_count`` seed the "files inserted"
    bar (0/0 -> an indeterminate importing row).
    """

    return ImportProbe(
        readiness=readiness,
        files_present=files_present,
        command_issued=command_issued,
        imported_count=imported_count,
        target_count=target_count,
    )


def manual_candidate(
    path: str,
    *,
    quality: QualityModel | None = None,
    rejections: list[Any] | None = None,
) -> ManualImportCandidate:
    """One parsed Sonarr manual-import candidate, as ``import_completed`` reads it.

    Mirrors the typed value ``SonarrClient.manual_import_candidates`` returns:
    only the fields the import decision consults are populated - ``path``
    (basename drives the episode-id lookup), the in-context ``quality`` fallback,
    and ``rejections`` (sample / already-imported skips). Built through
    ``ManualImportCandidate.from_api`` so the raw rejection shapes (a bare string
    or an ``{"reason": ...}`` dict) are folded exactly as in production.
    """

    return ManualImportCandidate.from_api(
        {"path": path, "quality": quality, "rejections": rejections or []},
    )


def make_sonarr_episodes(**attrs: Any) -> SonarrEpisodes:
    """A bare ``SonarrEpisodes`` with ``__init__`` bypassed and only ``attrs`` set.

    Mirrors ``make_sonarr_sync``: the per-run episode cache + series fingerprint
    default empty, so a test only sets the collaborators the method under test
    reads (``sonarr``, ``_services``, ``_config``, ``_mappings``, ``_anilist``, ...).
    """

    defaults: dict[str, Any] = {"_ep_list_cache": {}, "_series_fp": ""}
    defaults.update(attrs)
    return make_bare_instance(SonarrEpisodes, **defaults)


def make_sonarr_sync(
    *,
    sonarr: AbstractSonarrClient | None = None,
    config: AppConfig | None = None,
    cache_store: AbstractCacheStore | None = None,
    ep_list_cache: dict[int, list[SonarrEpisode]] | None = None,
) -> SonarrSync:
    """Build a ``SonarrSync`` through its REAL ``__init__``, injecting a typed client.

    Shrunk from the old hand-rebuilt collaborator graph (~12 private field names,
    zero type-checking): builds a real ``RunDeps`` + services hub and calls the real
    constructor, passing the scripted client through the typed ``sonarr_client``
    seam (default a blank ``FakeSonarrClient``). So the real two-phase wiring runs -
    the collaborators all share the injected client + the strat's ``cache_store`` by
    identity, exactly as production builds them - and a wrong/incomplete fake is a
    pyright error AND un-instantiable, not a silently-set dead attribute.
    ``ep_list_cache`` seeds the episode collaborator's per-run cache after
    construction (``__init__`` builds it empty), for the run-start-reset tests.
    """

    deps = make_run_deps(config=config, cache_store=cache_store)
    services = RunServices(deps, Arr.SONARR)
    strat = SonarrSync(
        deps,
        services,
        sonarr_client=sonarr if sonarr is not None else FakeSonarrClient(),
    )
    if ep_list_cache is not None:
        strat._episodes._ep_list_cache = ep_list_cache
    return strat


def make_sonarr_mapper(**attrs: Any) -> FileEpisodeMapper:
    """A bare ``FileEpisodeMapper`` with ``__init__`` bypassed and only ``attrs`` set.

    The tests assign just what the method under test reads (``sonarr`` for the
    on-disk ``/parse``); the per-run on-disk parse cache defaults empty.
    """

    defaults: dict[str, Any] = {"_parse_info_cache": {}}
    defaults.update(attrs)
    return make_bare_instance(FileEpisodeMapper, **defaults)


def make_sonarr_parse(**attrs: Any) -> SonarrParseCache:
    """A bare ``SonarrParseCache`` with ``__init__`` bypassed and only ``attrs`` set.

    The tests assign just what the method under test reads (``sonarr``, ``_config``,
    ``cache_store``, ``logger``). ``parse_episodes_from_seadex`` takes the run's
    ``series_fp`` as a call argument, so there is no per-run field to default here.
    """

    return make_bare_instance(SonarrParseCache, **attrs)
