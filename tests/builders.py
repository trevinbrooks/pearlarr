"""Builders and a bare-instance factory for the characterization tests.

These tests pin the *current* behaviour of ``seadex_arr.py``. The planner tests
build the engine's
inputs (typed episode records, flat release dicts) via the helpers here, and
``make_arr`` builds a
``SeaDexArr`` without running its heavy ``__init__`` (network downloads,
qBittorrent login, disk I/O), assigning only the attributes the methods under
test actually read.
"""

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any, cast
from unittest import mock

from seadex import EntryRecord, Tag, Tracker

from seadexarr.modules.cache import UPDATED_AT_STR_FORMAT, CachedEntry, CacheField, CacheRecord
from seadexarr.modules.config import AppConfig, Arr
from seadexarr.modules.grab_pipeline import GrabPipeline
from seadexarr.modules.import_wait import ImportWaitManager
from seadexarr.modules.manual_import import ImportProbe, ImportReadiness, ImportWaitMode, PendingImport
from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_filter import SeadexReleaseFilter
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    EpisodeRecord,
    ManualImportCandidate,
    QualityModel,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)
from seadexarr.modules.sonarr_episodes import SonarrEpisodes
from seadexarr.modules.sonarr_import import ImportExecutor, ImportReconciler
from seadexarr.modules.sonarr_mapper import FileEpisodeMapper
from seadexarr.modules.sonarr_parse import SonarrParseCache
from seadexarr.modules.torrents import AddOutcome

# Map each flat (group-local) setting name to its config group, derived straight from
# AppConfig's own field tree so it can't drift into a stale subset: adding a 9th
# settings group to AppConfig wires it in here for free. AppConfig declares ``sonarr``
# before ``radarr``, so a name shared across the two arr groups (the ArrSettings keys
# url/api_key/ignore_unmonitored/torrent_category) resolves to ``sonarr`` - the arr
# make_config/make_arr default to - via the first-wins ``setdefault`` below.
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
}


def _resolve_setting(key: str) -> tuple[str, str]:
    """Map a flat override key to its ``(group, field)`` in the nested config."""

    if key in _FLAT_ALIASES:
        return _FLAT_ALIASES[key]
    return _FIELD_GROUP.get(key, "seadex"), key


# The override keys make_arr routes into self._config (rather than onto the bare
# engine as a direct attribute/collaborator).
_CONFIG_SETTING_NAMES = frozenset(_FIELD_GROUP) | frozenset(_FLAT_ALIASES)


def make_bare_instance[T](cls: type[T], **attrs: Any) -> T:
    """An instance with ``__init__`` bypassed and only the given attrs set.

    ``object.__new__`` skips the real, heavy ``__init__`` (network downloads,
    qBittorrent login, disk I/O); the tests assign just the attributes the
    methods under test read. Shared by ``make_arr`` here and the strategy-seam
    tests so the bypass idiom lives in one place.
    """

    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


# The scalar entry columns ``update_cache`` merges, derived from the public
# ``CacheField`` vocabulary (every field but the hash set) so the fake can't drift
# from the real ``CacheStore``'s ``_ENTRY_SCALAR_COLUMNS``.
_FAKE_SCALAR_FIELDS: tuple[str, ...] = tuple(
    field.value for field in CacheField if field is not CacheField.TORRENT_HASHES
)


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


class FakeCacheStore:
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

    ``get_pending`` returns a fresh (outer) copy, matching the real snapshot
    semantics; ``save`` / ``close`` are no-ops; ``stats`` / ``integrity_check``
    report a plausible health snapshot. Arr keys use ``str(arr)`` to mirror
    production's ``_arr_key``.
    """

    def __init__(
        self,
        *,
        sonarr_parse: dict[str, dict[str, Any]] | None = None,
        pending: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._sonarr_parse: dict[str, dict[str, Any]] = dict(sonarr_parse or {})
        self._pending: dict[str, dict[str, dict[str, Any]]] = {
            str(arr): dict(recs) for arr, recs in (pending or {}).items()
        }
        # Per-entry records: the scalar columns keyed by (arr, al_id), and the
        # entry's torrent-hash set kept separately (the entries / torrent_hashes
        # split). An entry present with an empty scalar dict still "exists" - the
        # existence checks key on membership, never the dict's truthiness.
        self._entries: dict[tuple[str, int], dict[str, Any]] = {}
        self._entry_hashes: dict[tuple[str, int], list[str | None]] = {}
        self._anilist_meta: dict[int, dict[str, Any]] = {}

    # -- lifecycle --
    def save(self, *, preview: bool) -> None:
        del preview

    def close(self) -> None:
        pass

    # -- per-entry records (entries + torrent_hashes) --
    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> bool:
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
        return True

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

    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None:
        """The four scalar columns of the entry as a ``CachedEntry``, or None."""

        entry = self._entries.get((str(arr), al_id))
        if entry is None:
            return None
        return CachedEntry(
            updated_at=entry.get("updated_at"),
            name=entry.get("name"),
            url=entry.get("url"),
            coverage=entry.get("coverage"),
        )

    def get_cached_name(self, arr: Arr, al_id: int) -> str | None:
        return cast("str | None", self.get_cached_field(arr, al_id, CacheField.NAME))

    def get_cached_field(
        self,
        arr: Arr,
        al_id: int,
        field: CacheField,
    ) -> object | None:
        if field == CacheField.TORRENT_HASHES:
            return self.torrent_hashes(arr, al_id)
        entry = self._entries.get((str(arr), al_id))
        return None if entry is None else entry.get(field.value)

    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]:
        """The entry's hashes, ordered None-first then ascending (mirrors ORDER BY)."""

        stored = self._entry_hashes.get((str(arr), al_id), [])
        ordered: list[str | None] = [None] if None in stored else []
        ordered.extend(sorted(h for h in stored if h is not None))
        return ordered

    # -- AniList meta (TTL-swept) --
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        yield from list(self._anilist_meta.items())

    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        return self._anilist_meta.get(al_id)

    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None:
        self._anilist_meta[al_id] = record

    def evict_anilist_meta(self, cutoff: datetime) -> int:
        return _evict_stale(self._anilist_meta, cutoff)

    # -- Sonarr parse cache (TTL-swept) --
    def iter_sonarr_parse(self) -> Iterator[tuple[str, dict[str, Any]]]:
        yield from list(self._sonarr_parse.items())

    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        return self._sonarr_parse.get(filename)

    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None:
        self._sonarr_parse[filename] = record

    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        return _evict_stale(self._sonarr_parse, cutoff)

    # -- pending imports --
    def get_pending(self, arr: Arr) -> dict[str, dict[str, Any]]:
        return dict(self._pending.get(str(arr), {}))

    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]:
        """Fresh snapshot filtered to one series (mirrors the SQL ``->> 'series_id'``)."""

        return {
            infohash: record
            for infohash, record in self._pending.get(str(arr), {}).items()
            if record.get("series_id") == series_id
        }

    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None:
        self._pending.setdefault(str(arr), {})[infohash] = record

    def drop_pending(self, arr: Arr, infohash: str) -> None:
        self._pending.get(str(arr), {}).pop(infohash, None)

    # -- maintenance: stats, integrity --
    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "torrent_hashes": sum(len(h) for h in self._entry_hashes.values()),
            "anilist_meta": len(self._anilist_meta),
            "sonarr_parse": len(self._sonarr_parse),
            "pending_imports": sum(len(recs) for recs in self._pending.values()),
            "size_bytes": 0,
        }

    def integrity_check(self) -> str:
        return "ok"


def make_logger(name: str = "seadexarr-test") -> logging.Logger:
    """A quiet logger for the characterization tests.

    Attaches a NullHandler, disables propagation, and resets the level to
    WARNING on every call so the hot-path debug f-strings aren't formatted and a
    test that bumps the level can't leak into the next.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    return logger


def make_config(**overrides: Any) -> AppConfig:
    """An in-memory ``AppConfig`` carrying ``make_arr``'s decision-test defaults.

    The config flags are read through ``self._config`` (the single source of truth),
    so a bare instance needs a real ``AppConfig``. These defaults mirror the historical
    ``make_arr`` flags - notably ``public_only=False`` (``AppConfig``'s own default is
    True) - and leave ``trackers`` unset so it defaults to PUBLIC | PRIVATE. Each flat
    override is routed to its config group (``_FIELD_GROUP``) and the nested mapping is
    validated through the models, so the before-validators run exactly as on a real load.
    """

    nested: dict[str, dict[str, Any]] = {
        "seadex": {
            "public_only": False,
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


def make_arr(**overrides: Any) -> SeaDexArr:
    """Build a bare ``SeaDexArr`` with only the attributes the methods read.

    Bypasses ``__init__`` via ``object.__new__`` and assigns sane defaults for
    the collaborators the decision methods consult; the config flags live on an
    in-memory ``AppConfig`` (``self._config``). Pass keyword overrides to vary a
    single config flag (e.g. ``make_arr(public_only=True)``) or another attribute.

    ``SeaDexArr`` is a concrete engine after Phase 6b (no abstract hooks), so
    ``make_bare_instance`` builds one directly - the old no-op-hooks stub is gone.
    """

    logger = make_logger()

    # Config-backed flags are read via self._config after Phase 5b; route any
    # passed as overrides through an in-memory AppConfig, leaving the rest as
    # direct attributes/collaborators.
    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}

    config = make_config(**config_overrides)
    defaults: dict[str, Any] = {
        "logger": logger,
        "log_fmt": mock.MagicMock(),
        "_config": config,
        # The engine reads per-arr flags (e.g. ignore_unmonitored) off _arr_config,
        # the Sonarr view of the same shared config.
        "_arr_config": config.for_arr(Arr.SONARR),
        # The real __init__ always sets a RunContext; faithful default so methods
        # that read self._ctx.arr (the run-arr methods) work without each test
        # wiring one. Override with _ctx=... for a specific run state.
        "_ctx": RunContext(arr=Arr.SONARR),
    }
    defaults.update(overrides)
    return make_bare_instance(SeaDexArr, **defaults)


def make_release_filter(**overrides: Any) -> SeadexReleaseFilter:
    """Build a ``SeadexReleaseFilter`` with only what its methods read.

    Config-backed flags (e.g. ``want_best``, ``public_only``) route through an
    in-memory ``AppConfig``; the rest pass straight to the constructor. Mirrors
    ``make_arr``'s override routing so the ``build`` characterization tests read
    the same as the old ``get_seadex_dict`` ones.
    """

    logger = make_logger()
    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}
    config = make_config(**config_overrides)
    defaults: dict[str, Any] = {
        "config": config,
        "planner": make_planner(),
        "cache_store": FakeCacheStore(),
        "logger": logger,
        "log_fmt": mock.MagicMock(),
        "ctx": RunContext(arr=Arr.SONARR),
    }
    defaults.update(overrides)
    return SeadexReleaseFilter(**defaults)


# A truthy stand-in for a logged-in qBittorrent client, so _is_preview() is
# False without a real login (the actual add is faked by FakeTorrents).
CLIENT_SENTINEL = object()


class FakeTorrents:
    """Mimics ``TorrentService.add``: a per-hash scripted ``(outcome, name)``.

    Keyed by infohash (not call order) so a multi-release add can return a
    different outcome per release regardless of dict iteration order.
    """

    def __init__(self, by_hash: dict[str | None, tuple[AddOutcome, str | None]]) -> None:
        self._by_hash = by_hash
        self.calls: list[str | None] = []

    def add(
        self,
        *,
        url: str,
        tracker: object,
        torrent_hash: str | None,
        preview: bool,
    ) -> tuple[AddOutcome, str | None]:
        del url, tracker, preview
        self.calls.append(torrent_hash)
        return self._by_hash[torrent_hash]


def one_release_dict(*, srg: str, infohash: str, url: str = "https://nyaa.si/view/1") -> dict:
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

    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}
    config = make_config(**config_overrides)
    notifier = mock.MagicMock()
    notifier.enabled = False
    defaults: dict[str, Any] = {
        "_config": config,
        "_planner": make_planner(),
        "cache_store": FakeCacheStore(),
        "_torrents": mock.MagicMock(),
        "_anilist": mock.MagicMock(),
        "_notifier": notifier,
        "_reporter": mock.MagicMock(),
        "log_fmt": mock.MagicMock(),
        "qbit": CLIENT_SENTINEL,
        "_ctx": RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING),
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

    config_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _CONFIG_SETTING_NAMES}
    config = make_config(**config_overrides)
    defaults: dict[str, Any] = {
        "_config": config,
        "cache_store": FakeCacheStore(),
        "_reporter": mock.MagicMock(),
        "logger": make_logger(),
        "qbit": None,
        "_ctx": RunContext(arr=Arr.SONARR),
        "_active_strategy": mock.MagicMock(),
    }
    defaults.update(overrides)
    return make_bare_instance(ImportWaitManager, **defaults)


def make_planner(**overrides: Any) -> DownloadPlanner:
    """Build a ``DownloadPlanner`` with test-friendly defaults.

    The planner reads three config flags plus a logger; pass keyword overrides
    to vary a single flag (e.g. ``make_planner(public_only=True)``). The logger
    defaults to WARNING so the hot-path debug f-strings aren't formatted, mirroring
    ``make_arr``.
    """

    logger = make_logger()

    defaults: dict[str, Any] = {
        "public_only": False,
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
    is_public: bool = True,
    infohash: str | None = "hash1",
    download: bool = False,
    episodes: list[EpisodeRecord] | None = None,
) -> SeadexUrlItem:
    """One SeaDex URL record, matching ``get_seadex_dict``'s ``url_item`` shape."""

    return SeadexUrlItem(
        url=url,
        files=files or [],
        size=size or [],
        is_public=is_public,
        hash=infohash,
        download=download,
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


class FakeTracker:
    """Mimics a seadex ``Tracker``: has ``casefold()`` and ``is_public()``."""

    def __init__(self, name: str, public: bool) -> None:
        self.name = name
        self._public = public

    def casefold(self) -> str:
        return self.name.casefold()

    def lower(self) -> str:
        return self.name.lower()

    def is_public(self) -> bool:
        return self._public


class FakeFile:
    """Mimics a seadex torrent file (``name`` / ``size``)."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


class FakeTorrent:
    """Mimics a seadex torrent record (the fields ``get_seadex_dict`` reads)."""

    def __init__(
        self,
        *,
        release_group: str,
        url: str,
        tracker: FakeTracker,
        files: list[FakeFile] | None = None,
        tags: list[str] | None = None,
        is_best: bool = False,
        is_dual_audio: bool = False,
        infohash: str | None = "hash",
    ) -> None:
        self.release_group = release_group
        self.url = url
        self.tracker = tracker
        self.files = files or []
        self.tags = tags or []
        self.is_best = is_best
        self.is_dual_audio = is_dual_audio
        self.infohash = infohash


class FakeEntry:
    """Mimics a seadex ``EntryRecord`` for ``get_seadex_dict`` (reads ``.torrents``)."""

    def __init__(self, torrents: list[FakeTorrent]) -> None:
        self.torrents = torrents


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
        "season_number": 1,
        "seadex_files": ["Show - 01 [1080p].mkv"],
        "seadex_sizes": [1000],
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
) -> ImportProbe:
    """An :class:`ImportProbe` with the common "files verified imported" defaults.

    The default is the verified-import outcome (``IMPORTED`` + ``files_present``);
    pass ``readiness=RETRY, files_present=False, command_issued=True`` for the
    "command accepted, copy in flight" case, or ``files_present=False`` for a not-
    yet-ready poll.
    """

    return ImportProbe(
        readiness=readiness,
        files_present=files_present,
        command_issued=command_issued,
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


def make_sonarr_sync(**attrs: Any) -> SonarrSync:
    """A bare ``SonarrSync`` with ``__init__`` bypassed and only ``attrs`` set.

    Mirrors ``make_arr`` / ``make_bare_instance``: no live Sonarr client is built,
    so the tests assign just the collaborators the method under test reads
    (``sonarr``, ``logger``, ``_config``, and the per-run caches).

    The episode-domain state (``_ep_list_cache`` / ``_series_fp``) lives on the
    ``SonarrEpisodes`` collaborator now, and the four per-run import caches
    (``_quality_defs_cache`` / ``_languages_cache`` / ``_warned_unplaceable`` /
    ``_last_refresh_monotonic``) on the ``ImportExecutor`` collaborator, exactly as
    the real ``__init__`` builds them - so a test still passes those field names and
    they are routed to the right collaborator (the executor sharing the strat's
    mapper + client/config/logger). Pass ``_episodes=`` / ``_executor=`` to override
    a whole collaborator.
    """

    defaults: dict[str, Any] = {}
    defaults.update(attrs)
    episodes = defaults.pop("_episodes", None)
    parse = defaults.pop("_parse", None)
    mapper = defaults.pop("_mapper", None)
    executor = defaults.pop("_executor", None)
    reconciler = defaults.pop("_reconciler", None)
    ep_cache = defaults.pop("_ep_list_cache", {})
    series_fp = defaults.pop("_series_fp", "")
    parse_info = defaults.pop("_parse_info_cache", {})
    quality_defs = defaults.pop("_quality_defs_cache", None)
    languages = defaults.pop("_languages_cache", None)
    warned = defaults.pop("_warned_unplaceable", set())
    last_refresh = defaults.pop("_last_refresh_monotonic", None)
    strat = make_bare_instance(SonarrSync, **defaults)
    if episodes is None:
        episodes = make_sonarr_episodes(
            sonarr=defaults.get("sonarr"),
            _services=defaults.get("_services"),
            _config=defaults.get("_config"),
            _mappings=defaults.get("_mappings"),
            anibridge=defaults.get("anibridge"),
            _anilist=defaults.get("_anilist"),
            log_fmt=defaults.get("log_fmt"),
            _ep_list_cache=ep_cache,
            _series_fp=series_fp,
        )
    strat._episodes = episodes
    if parse is None:
        # Share the SAME cache_store as the strat so a parse write is visible to
        # the seed builder's read later in the run (the staged-write invariant).
        parse = make_sonarr_parse(
            sonarr=defaults.get("sonarr"),
            _config=defaults.get("_config"),
            cache_store=defaults.get("cache_store"),
            logger=defaults.get("logger"),
        )
    strat._parse = parse
    if mapper is None:
        # Same sonarr client as the strat so import_completed's on-disk /parse hits
        # the test's scripted mock; routes the per-run on-disk parse cache here.
        mapper = make_sonarr_mapper(
            sonarr=defaults.get("sonarr"),
            _parse_info_cache=parse_info,
        )
    strat._mapper = mapper
    if executor is None:
        # Share the strat's mapper + client/config/logger so the import_completed-
        # driven tests exercise the same objects (and the same on-disk /parse mock);
        # routes the four per-run import caches here.
        executor = make_import_executor(
            sonarr=defaults.get("sonarr"),
            _config=defaults.get("_config"),
            logger=defaults.get("logger"),
            _mapper=mapper,
            _quality_defs_cache=quality_defs,
            _languages_cache=languages,
            _warned_unplaceable=warned,
            _last_refresh_monotonic=last_refresh,
        )
    strat._executor = executor
    if reconciler is None:
        # Composes the strat's episodes + executor, sharing the same cache_store so
        # build_pending_seeds reads back the parse writes (the staged-write invariant).
        reconciler = make_import_reconciler(
            _episodes=episodes,
            _executor=executor,
            cache_store=defaults.get("cache_store"),
            logger=defaults.get("logger"),
        )
    strat._reconciler = reconciler
    return strat


def make_import_executor(**attrs: Any) -> ImportExecutor:
    """A bare ``ImportExecutor`` with ``__init__`` bypassed and only ``attrs`` set.

    The tests assign just what the method under test reads (``sonarr`` for the
    manual-import calls, ``_mapper`` for the file->episode map, ``_config`` /
    ``logger``); the four per-run caches default to their run-start values (None /
    empty) so the lazy-fetch + warn-once paths run unless a test pre-seeds them.
    """

    defaults: dict[str, Any] = {
        "_quality_defs_cache": None,
        "_languages_cache": None,
        "_warned_unplaceable": set(),
        "_last_refresh_monotonic": None,
    }
    defaults.update(attrs)
    return make_bare_instance(ImportExecutor, **defaults)


def make_import_reconciler(**attrs: Any) -> ImportReconciler:
    """A bare ``ImportReconciler`` with ``__init__`` bypassed and only ``attrs`` set.

    The reconciler holds no per-run caches; it composes ``_episodes`` + ``_executor``
    and reads ``cache_store`` / ``logger``, so a test sets just those (the tests
    mostly drive it through the strat's import_completed / build_pending_seeds).
    """

    return make_bare_instance(ImportReconciler, **attrs)


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
