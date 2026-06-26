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
from typing import Any
from unittest import mock

from seadex import Tag

from seadexarr.modules.config import AppConfig, Arr
from seadexarr.modules.manual_import import ImportProbe, ImportReadiness, PendingImport
from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    EpisodeRecord,
    ManualImportCandidate,
    QualityModel,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)

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
    config_overrides = {
        key: overrides.pop(key)
        for key in list(overrides)
        if key in _CONFIG_SETTING_NAMES
    }

    config = make_config(**config_overrides)
    defaults: dict[str, Any] = {
        "logger": logger,
        "log_fmt": mock.MagicMock(),
        "_config": config,
        # The engine reads per-arr flags (e.g. ignore_unmonitored) off _arr_config,
        # the Sonarr view of the same shared config.
        "_arr_config": config.for_arr(Arr.SONARR),
    }
    defaults.update(overrides)
    return make_bare_instance(SeaDexArr, **defaults)


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


def make_sonarr_sync(**attrs: Any) -> SonarrSync:
    """A bare ``SonarrSync`` with ``__init__`` bypassed and only ``attrs`` set.

    Mirrors ``make_arr`` / ``make_bare_instance``: no live Sonarr client is built,
    so the tests assign just the collaborators the method under test reads
    (``sonarr``, ``logger``, ``_config``, and the per-run caches). The two
    per-run quality/language caches default to None (not yet fetched) so the
    lazy-fetch path runs unless a test pre-seeds them.
    """

    defaults: dict[str, Any] = {
        "_quality_defs_cache": None,
        "_languages_cache": None,
    }
    defaults.update(attrs)
    return make_bare_instance(SonarrSync, **defaults)
