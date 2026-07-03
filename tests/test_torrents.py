# pyright: strict
"""Tests for ``TorrentService`` (the qBittorrent add path) and the
``APIConnectionError`` -> ``QbitConnectionError`` mapping.

``TorrentService.add`` always runs the tracker parser then ``_add_to_qbit``.
These tests monkeypatch the tracker parsers (the HTML/feed scrapers, covered in
``test_torrent_parsers``) so each test drives only the qbit side, against a typed
recording fake qBittorrent client built inline here. ``qbittorrentapi.Client`` is
a concrete SDK class with no Protocol seam, so the fake is cast to it at the one
injection boundary (the sanctioned leaf-client hatch).

The ``QbitConnectionError`` mapping itself lives in ``RunDeps.build`` (where the
login happens), so that one path is exercised through ``build`` with the qbit
``Client`` swapped for one whose ``auth_log_in`` raises.
"""

import logging
from dataclasses import dataclass
from typing import cast

import pytest
import qbittorrentapi
import requests
from seadex import Tracker

import seadexarr.modules.torrents as torrents
from seadexarr.modules.config import Arr
from seadexarr.modules.mappings import MappingResolver
from seadexarr.modules.run_services import QbitConnectionError, RunDeps
from seadexarr.modules.torrent import TorrentParseError
from seadexarr.modules.torrents import PARSEABLE_TRACKERS, AddOutcome, TorrentAddError, TorrentService

from .builders import make_bare_instance, make_config

_HASH = "a" * 40
_PARSED_URL = "magnet:?xt=urn:btih:DEAD"
_SOURCE_TITLE = "Scraped Source Title"


@dataclass(frozen=True)
class _TorrInfo:
    """The ``torrents_info`` row the add path reads (``.hash`` / ``.name``)."""

    hash: str
    name: str


@dataclass(frozen=True)
class _AddCall:
    """One recorded ``torrents_add`` invocation (the request shape)."""

    urls: str
    category: str | None
    tags: list[str] | None


class _FakeQbit:
    """A typed recording stand-in for ``qbittorrentapi.Client``.

    Models qBittorrent's dedup-by-hash: ``torrents_info`` reports whichever hashes
    are ``present``; ``torrents_add`` records its call and (optionally) registers a
    hash->name as now-present, so the post-add name read-back finds it.
    """

    def __init__(
        self,
        *,
        present: dict[str, str] | None = None,
        register_on_add: tuple[str, str] | None = None,
        add_result: str = "Ok.",
    ) -> None:
        self._present: dict[str, str] = dict(present or {})
        self._register_on_add = register_on_add
        self._add_result = add_result
        self.add_calls: list[_AddCall] = []
        self.info_queries: list[str] = []

    def torrents_info(self, *, torrent_hashes: str) -> list[_TorrInfo]:
        self.info_queries.append(torrent_hashes)
        name = self._present.get(torrent_hashes)
        return [_TorrInfo(hash=torrent_hashes, name=name)] if name is not None else []

    def torrents_add(self, *, urls: str, category: str | None, tags: list[str] | None) -> str:
        self.add_calls.append(_AddCall(urls=urls, category=category, tags=tags))
        if self._register_on_add is not None:
            registered_hash, registered_name = self._register_on_add
            self._present[registered_hash] = registered_name
        return self._add_result


def _service(qbit: _FakeQbit, *, category: str | None = "anime", tags: list[str] | None = None) -> TorrentService:
    """A ``TorrentService`` over the fake qbit (cast at the leaf-client boundary)."""

    return TorrentService(
        qbit=cast("qbittorrentapi.Client", qbit),
        session=requests.Session(),
        category=category,
        tags=tags if tags is not None else ["seadex"],
        logger=logging.getLogger("seadexarr.test"),
    )


def _patch_nyaa_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive only the qbit side: the Nyaa parser returns a fixed (url, title)."""

    def _fixed_nyaa(url: str) -> tuple[str, str]:
        del url
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _fixed_nyaa)


# --- TorrentService.add (non-preview qbit path) -----------------------------


def test_add_new_torrent_with_hash_prefers_qbit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh add hands the parsed URL to ``torrents_add`` and reports the name
    qBittorrent reads back (which wins over the scraped source title).
    """

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(register_on_add=(_HASH, "Qbit Reported Name"))
    service = _service(qbit, category="anime", tags=["seadex"])

    outcome, name = service.add(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH, preview=False)

    assert outcome is AddOutcome.ADDED
    assert name == "Qbit Reported Name"
    assert qbit.add_calls == [_AddCall(urls=_PARSED_URL, category="anime", tags=["seadex"])]
    # info queried twice: the pre-add dedup scan, then the post-add name read-back.
    assert qbit.info_queries == [_HASH, _HASH]


def test_add_already_present_dedups_by_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the hash is already in qBittorrent, the add is skipped and reported as
    ALREADY_ADDED with the existing name.
    """

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(present={_HASH: "Existing Name"})
    service = _service(qbit)

    outcome, name = service.add(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH, preview=False)

    assert outcome is AddOutcome.ALREADY_ADDED
    assert name == "Existing Name"
    assert qbit.add_calls == []


def test_add_hashless_falls_back_to_source_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hashless torrent skips the dedup/name lookup, so the reported name falls
    back to the scraped source title.
    """

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit()
    service = _service(qbit)

    outcome, name = service.add(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=None, preview=False)

    assert outcome is AddOutcome.ADDED
    assert name == _SOURCE_TITLE
    assert qbit.add_calls == [_AddCall(urls=_PARSED_URL, category="anime", tags=["seadex"])]
    assert qbit.info_queries == []


def test_add_qbit_rejects_add_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-``"Ok."`` ``torrents_add`` result is a hard failure."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(add_result="Fails.")
    service = _service(qbit)

    with pytest.raises(TorrentAddError, match="Failed to add torrent"):
        service.add(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=None, preview=False)


def test_add_unparseable_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the tracker parser yields no download URL, the add raises before qbit."""

    def _none_animetosho(url: str, session: requests.Session) -> tuple[str | None, str]:
        del url, session
        return (None, "Title")

    monkeypatch.setattr(torrents, "get_animetosho_torrent", _none_animetosho)
    qbit = _FakeQbit()
    service = _service(qbit)

    with pytest.raises(TorrentParseError, match="Have not managed to parse the torrent URL"):
        service.add(url="https://animetosho.org/view/1", tracker=Tracker.ANIMETOSHO, infohash=None, preview=False)

    assert qbit.add_calls == []


def _patch_all_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub all three tracker parsers to a fixed (url, title), so the parseable
    branch of ``add`` reaches the qbit path without a real scrape."""

    def _nyaa(url: str) -> tuple[str | None, str]:
        del url
        return (_PARSED_URL, _SOURCE_TITLE)

    def _animetosho(url: str, session: requests.Session) -> tuple[str | None, str]:
        del url, session
        return (_PARSED_URL, _SOURCE_TITLE)

    def _rutracker(url: str, infohash: str | None, session: requests.Session) -> tuple[str | None, str]:
        del url, infohash, session
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _nyaa)
    monkeypatch.setattr(torrents, "get_animetosho_torrent", _animetosho)
    monkeypatch.setattr(torrents, "get_rutracker_torrent", _rutracker)


@pytest.mark.parametrize("tracker", list(Tracker))
def test_add_raises_iff_tracker_unparseable(tracker: Tracker, monkeypatch: pytest.MonkeyPatch) -> None:
    """``add`` parses exactly the ``PARSEABLE_TRACKERS`` and raises for every other
    member - pinning the constant against ``add``'s dispatch dict so drift in either
    direction (a parser added without the constant, or vice-versa) is caught."""

    _patch_all_parsers(monkeypatch)
    service = _service(_FakeQbit())

    if tracker in PARSEABLE_TRACKERS:
        outcome, _ = service.add(url="https://example/1", tracker=tracker, infohash=None, preview=True)
        assert outcome is AddOutcome.ADDED
    else:
        with pytest.raises(ValueError, match="Unable to parse torrent links"):
            service.add(url="https://example/1", tracker=tracker, infohash=None, preview=True)


# --- QbitConnectionError mapping (RunDeps.build login path) ------------------


def test_qbit_login_failure_maps_to_qbit_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A qBittorrent ``APIConnectionError`` at login is remapped to the
    user-facing ``QbitConnectionError`` by ``RunDeps.build``.
    """

    class _FailingClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def auth_log_in(self) -> None:
            raise qbittorrentapi.APIConnectionError("login failed")

    monkeypatch.setattr(qbittorrentapi, "Client", _FailingClient)

    config = make_config(host="http://qbit:8080", username="user", password="pass")
    # Guard: the kwargs must route to the qbittorrent group, else build skips qbit.
    assert config.qbittorrent.credentials() is not None

    with pytest.raises(QbitConnectionError):
        RunDeps.build(
            Arr.SONARR,
            app_config=config,
            logger=logging.getLogger("seadexarr.test"),
            mappings=make_bare_instance(MappingResolver),
        )
