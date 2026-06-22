"""Builders and a bare-instance factory for the characterization tests.

These tests pin the *current* behaviour of ``seadex_arr.py`` before the planned
decomposition (see ``REFACTOR_PLAN.md``). The decision engine operates on plain
dicts, so the planner tests build those dicts directly. ``make_arr`` builds a
``SeaDexArr`` without running its heavy ``__init__`` (network downloads,
qBittorrent login, disk I/O), assigning only the attributes the methods under
test actually read.
"""

import logging
from typing import Any
from unittest import mock

from seadexarr.modules.seadex_arr import (
    PRIVATE_TRACKERS,
    PUBLIC_TRACKERS,
    SeaDexArr,
)


class _StubArr(SeaDexArr):
    """A concrete ``SeaDexArr`` whose abstract hooks are no-ops.

    Exists only so ``object.__new__`` can build an instance: ``object.__new__``
    refuses a class that still has abstract methods, so all four hooks are
    implemented here even though the characterization tests never call them.
    """

    def _get_all_items(self) -> list:
        return []

    def _filter_to_single_item(self, items: list, item_id: int) -> list:
        del item_id
        return items

    def _item_anilist_ids(self, item: Any, log_ignored: bool = True) -> dict:
        del item, log_ignored
        return {}

    def _process_al_id(
        self,
        arr: str,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: dict,
    ) -> bool:
        del arr, item, item_title, al_id, mapping
        return False


def make_arr(**overrides: Any) -> _StubArr:
    """Build a bare ``SeaDexArr`` with only the attributes the methods read.

    Bypasses ``__init__`` via ``object.__new__`` and assigns sane defaults for
    the config flags / collaborators the decision methods consult. Pass keyword
    overrides to vary a single flag (e.g. ``make_arr(public_only=True)``).
    """

    arr = object.__new__(_StubArr)

    logger = logging.getLogger("seadexarr-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    # Default to WARNING so the hot-path debug f-strings aren't formatted and
    # the ``debug_on`` branch is exercised in its common (off) state. Reset on
    # every call, so a test that bumps the level can't leak into the next.
    logger.setLevel(logging.WARNING)

    defaults: dict[str, Any] = {
        "logger": logger,
        "log_fmt": mock.MagicMock(),
        "interactive": False,
        "public_only": False,
        "want_best": True,
        "prefer_dual_audio": True,
        "ignore_tags": [],
        "trackers": PUBLIC_TRACKERS | PRIVATE_TRACKERS,
        "public_only_skipped": False,
        "public_only_groups": [],
        "cache": {"anilist_entries": {}},
        "use_torrent_hash_to_filter": False,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(arr, name, value)
    return arr


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
