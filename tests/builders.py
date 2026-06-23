"""Builders and a bare-instance factory for the characterization tests.

These tests pin the *current* behaviour of ``seadex_arr.py`` before the planned
decomposition (see ``REFACTOR_PLAN.md``). The decision engine operates on plain
dicts, so the planner tests build those dicts directly. ``make_arr`` builds a
``SeaDexArr`` without running its heavy ``__init__`` (network downloads,
qBittorrent login, disk I/O), assigning only the attributes the methods under
test actually read.
"""

import logging
from functools import cached_property
from typing import Any, TypeVar
from unittest import mock

from seadexarr.modules.config import AppConfig
from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.seadex_arr import SeaDexArr

# The override keys make_arr routes into self._config, derived from AppConfig's
# real setting surface so it can't drift into a stale subset. The old hardcoded
# list silently misrouted any flag it omitted to a dead direct attribute that no
# code reads (so the test exercised the config default while looking like it set
# the override).
_CONFIG_SETTING_NAMES = frozenset(
    name
    for name, attr in vars(AppConfig).items()
    if isinstance(attr, (property, cached_property))
)


T = TypeVar("T")


def make_bare_instance(cls: type[T], **attrs: Any) -> T:
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

    After Phase 5b the config flags are read through ``self._config`` (the single
    source of truth), so a bare instance needs a real ``AppConfig`` rather than
    flat mirror attributes. These defaults mirror the historical ``make_arr``
    flags - notably ``public_only=False`` (``AppConfig``'s own default is True) -
    and leave ``trackers`` unset so it defaults to PUBLIC | PRIVATE.
    """

    data: dict[str, Any] = {
        "public_only": False,
        "want_best": True,
        "prefer_dual_audio": True,
        "ignore_tags": [],
        "interactive": False,
        "use_torrent_hash_to_filter": False,
    }
    # AppConfig reads a few settings under an ``{arr}_`` data key (e.g.
    # ``ignore_unmonitored`` -> ``sonarr_ignore_unmonitored``). Write each
    # override under both the bare and the sonarr-prefixed key so it takes effect
    # whichever form the property reads - no per-key allow-list to keep in sync.
    for key, value in overrides.items():
        data[key] = value
        data[f"sonarr_{key}"] = value
    return AppConfig(path="unused.yml", arr="sonarr", data=data)


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

    defaults: dict[str, Any] = {
        "logger": logger,
        "log_fmt": mock.MagicMock(),
        "_config": make_config(**config_overrides),
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
    tracker: str = "Nyaa",
    is_public: bool = True,
    infohash: str | None = "hash1",
    download: bool = False,
    episodes: list[dict] | None = None,
) -> dict:
    """One SeaDex URL record, matching ``get_seadex_dict``'s ``url_item`` shape."""

    return {
        "url": url,
        "files": files or [],
        "size": size or [],
        "tracker": tracker,
        "is_public": is_public,
        "hash": infohash,
        "download": download,
        "episodes": episodes or [],
    }


def rg_group(
    urls: dict[str, dict],
    *,
    tags: list[str] | None = None,
    all_episodes: list[dict] | None = None,
) -> dict:
    """One SeaDex release-group record keyed by url.

    ``all_episodes`` is only set when provided, so the three branches of
    ``get_same_files_groups`` (absent -> no-parsing, ``[]`` -> unparsed,
    populated -> coverage frozenset) can each be reached.
    """

    group: dict[str, Any] = {"urls": urls, "tags": tags or []}
    if all_episodes is not None:
        group["all_episodes"] = all_episodes
    return group


def sonarr_ep(
    season: int,
    episode: int,
    *,
    size: int | None = None,
    release_group: str | None = None,
    episode_file_id: int = 1,
) -> dict:
    """One Sonarr episode dict, matching the fields the engine reads."""

    return {
        "seasonNumber": season,
        "episodeNumber": episode,
        "episodeFileId": episode_file_id,
        "episodeFile": {"size": size, "releaseGroup": release_group},
    }


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
