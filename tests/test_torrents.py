# pyright: strict
"""Tests for `TorrentService` (the add path and the cross-seed marks) and the `QbitConnectionError` mapping.

`TorrentService.add` asks qBittorrent for the hash, then runs the tracker parser and `_add_to_qbit`.
These tests monkeypatch the tracker parsers (the HTML/feed scrapers, covered in
`test_torrent_parsers`) so each test drives only the qbit side, against a typed
recording fake qBittorrent client built inline here. `qbittorrentapi.Client` is
a concrete SDK class with no Protocol seam, so the fake is cast to it at the one
injection boundary (the sanctioned leaf-client hatch).

The `QbitConnectionError` mapping itself lives in `RunDeps.build` (where the
login happens), so that one path is exercised through `build` with the qbit
`Client` swapped for one whose `auth_log_in` raises.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import cast

import httpx
import pytest
import qbittorrentapi
import respx
from seadex import Tracker

import pearlarr.torrents as torrents
from pearlarr.arr_categories import ArrCategoryResolver, CategorySources
from pearlarr.boot_flow import BootFlow
from pearlarr.config import Arr, ArrSettings
from pearlarr.manual_import import ImportWaitMode
from pearlarr.mappings import MappingResolver
from pearlarr.output import Severity
from pearlarr.reporter import RunContext
from pearlarr.run_services import QbitConnectionError, RunDeps, RunServices
from pearlarr.seadex_types import SeadexReleaseGroupItem, SeadexUrlItem
from pearlarr.torrent import TorrentParseError
from pearlarr.torrents import (
    GRAB_FAILURES,
    PARSEABLE_TRACKERS,
    AddOutcome,
    TorrentAddError,
    TorrentService,
)

from .builders import (
    download_client_json,
    make_bare_instance,
    make_categories,
    make_config,
    make_entry_record,
    make_run_deps,
    make_torrent_record,
    sonarr_client_fields,
    url_item,
)
from .fakes import bind_arr_http, diagnostic_messages, install_recording_hub

_HASH = "a" * 40
_PARSED_URL = "magnet:?xt=urn:btih:DEAD"
_SOURCE_TITLE = "Scraped Source Title"
_GRAB = "anime"
_POST = "anime-done"


@dataclass(frozen=True)
class _TorrInfo:
    """A `torrents_info` row: the fields the add path and the cross-seed look read."""

    hash: str
    name: str
    added_on: object = None
    category: str = _GRAB
    total_size: int = 0
    state: str = "downloading"
    progress: float = 0.0

    @property
    def state_enum(self) -> qbittorrentapi.TorrentState:
        return qbittorrentapi.TorrentState(self.state)


@dataclass(frozen=True)
class _TorrFile:
    """A `torrents_files` row: the file's path inside the torrent, its size, and its priority (0 is not downloaded)."""

    name: str
    size: int
    priority: int = 1


@dataclass(frozen=True)
class _AddCall:
    """One recorded `torrents_add` invocation (the request shape)."""

    urls: str
    category: str | None
    tags: list[str] | None


class _FakeQbit:
    """A typed recording stand-in for `qbittorrentapi.Client`.

    Models qBittorrent's dedup-by-hash: `torrents_info` reports whichever hashes
    are `present` or in `rows`, and lists a category's `rows` (`list_error` fails the list). `torrents_add`
    records its call and (optionally) registers a hash->name as now-present, so the post-add name read-back finds it.
    `torrents_files` answers each torrent's `files`, raising a scripted error, or 404 for a torrent with none.
    """

    def __init__(
        self,
        *,
        present: dict[str, str] | None = None,
        rows: tuple[_TorrInfo, ...] = (),
        files: Mapping[str, list[_TorrFile] | Exception] | None = None,
        list_error: Exception | None = None,
        register_on_add: tuple[str, str] | None = None,
        add_result: str = "Ok.",
        added_on: object = None,
    ) -> None:
        self._rows = {h: _TorrInfo(h, n, added_on) for h, n in (present or {}).items()} | {r.hash: r for r in rows}
        self._files = files or {}
        self._list_error = list_error
        self._added_on = added_on
        self._register_on_add = register_on_add
        self._add_result = add_result
        self.add_calls: list[_AddCall] = []
        self.info_queries: list[str] = []
        self.category_lists: list[str] = []
        self.files_reads: list[str] = []

    def torrents_info(self, *, torrent_hashes: str = "", category: str | None = None) -> list[_TorrInfo]:
        if category is not None:
            self.category_lists.append(category)
            if self._list_error is not None:
                raise self._list_error
            return [row for row in self._rows.values() if row.category == category]
        self.info_queries.append(torrent_hashes)
        # qBittorrent treats an empty `hashes` filter as "no filter" and returns
        # every torrent - the footgun the empty-infohash normalization guards against.
        if not torrent_hashes:
            return list(self._rows.values())
        row = self._rows.get(torrent_hashes)
        return [row] if row is not None else []

    def torrents_files(self, *, torrent_hash: str) -> list[_TorrFile]:
        self.files_reads.append(torrent_hash)
        files = self._files.get(torrent_hash, qbittorrentapi.NotFound404Error())
        if isinstance(files, Exception):
            raise files
        return files

    def torrents_add(self, *, urls: str, category: str | None, tags: list[str] | None) -> str:
        self.add_calls.append(_AddCall(urls=urls, category=category, tags=tags))
        if self._register_on_add is not None:
            registered_hash, registered_name = self._register_on_add
            self._rows[registered_hash] = _TorrInfo(registered_hash, registered_name, self._added_on)
        return self._add_result


def _service(
    qbit: _FakeQbit,
    *,
    category: str | None = _GRAB,
    tags: list[str] | None = None,
    categories: ArrCategoryResolver | None = None,
) -> TorrentService:
    """A `TorrentService` over the fake qbit (cast at the leaf-client boundary).

    `category` rides a fetchless resolver (no transport), so the explicit
    value passes through to the add exactly as configured. The post-import category is `_POST`.
    `categories` swaps in a resolver that reads the arr's clients.
    """

    config = make_config(sonarr_torrent_category=category, sonarr_post_import_category=_POST)
    return TorrentService(
        qbit=cast("qbittorrentapi.Client", qbit),
        web=httpx.Client(),
        categories=categories or make_categories(config),
        tags=tags if tags is not None else ["seadex"],
        logger=logging.getLogger("pearlarr.test"),
    )


def _patch_nyaa_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive only the qbit side: the Nyaa parser returns a fixed (url, title)."""

    def _fixed_nyaa(url: str, client: httpx.Client) -> tuple[str, str]:
        del url, client
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _fixed_nyaa)


# --- TorrentService.add (non-preview qbit path) -----------------------------


def test_add_new_torrent_with_hash_prefers_qbit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh add hands the parsed URL to `torrents_add` and reports the name qBittorrent reads back (which wins over the scraped source title)."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(register_on_add=(_HASH, "Qbit Reported Name"))
    service = _service(qbit, category="anime", tags=["seadex"])

    outcome, name, _ = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH), preview=False
    )

    assert outcome is AddOutcome.ADDED
    assert name == "Qbit Reported Name"
    assert qbit.add_calls == [_AddCall(urls=_PARSED_URL, category="anime", tags=["seadex"])]
    # info queried twice: the pre-add dedup scan, then the post-add name read-back.
    assert qbit.info_queries == [_HASH, _HASH]


def test_add_already_present_dedups_by_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hash already in qBittorrent is ALREADY_ADDED with the existing name, and its tracker page is never fetched."""

    def _page_gone(url: str, client: httpx.Client) -> tuple[str, str]:
        del client
        raise TorrentParseError(f"page gone: {url}")

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _page_gone)
    qbit = _FakeQbit(present={_HASH: "Existing Name"})
    service = _service(qbit)

    outcome, name, _ = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH), preview=False
    )

    assert outcome is AddOutcome.ALREADY_ADDED
    assert name == "Existing Name"
    assert qbit.add_calls == []


def test_add_hashless_falls_back_to_source_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hashless torrent skips the dedup/name lookup, so the reported name falls back to the scraped source title."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit()
    service = _service(qbit)

    outcome, name, _ = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=None), preview=False
    )

    assert outcome is AddOutcome.ADDED
    assert name == _SOURCE_TITLE
    assert qbit.add_calls == [_AddCall(urls=_PARSED_URL, category="anime", tags=["seadex"])]
    assert qbit.info_queries == []


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("   ", None), (_HASH, _HASH), (f"  {_HASH}  ", _HASH)])
def test_seadex_url_item_normalizes_infohash(raw: str, expected: str | None) -> None:
    """`SeadexUrlItem` strips its infohash and folds an empty/blank one to None (a real hash is never empty)."""

    assert SeadexUrlItem(infohash=raw).infohash == expected


def test_add_empty_infohash_is_treated_as_hashless(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string infohash is grabbed, not falsely deduped against every torrent.

    qBittorrent returns *all* torrents for an empty `hashes` filter, so a bare
    truthiness dedup would report ALREADY_ADDED against an unrelated torrent.
    `SeadexUrlItem` folds "" -> None at construction, so the dedup lookup is
    skipped and the release is added.
    """

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(present={_HASH: "Unrelated Torrent"})
    service = _service(qbit)

    outcome, name, _ = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=""), preview=False
    )

    assert outcome is AddOutcome.ADDED
    assert name == _SOURCE_TITLE
    assert qbit.info_queries == []  # dedup lookup skipped: "" was normalized to None


def test_add_qbit_rejects_add_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-`"Ok."` `torrents_add` result is a hard failure."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(add_result="Fails.")
    service = _service(qbit)

    # The message names the release URL and qBittorrent's response (it surfaces
    # verbatim in the pipeline's containment warning).
    with pytest.raises(TorrentAddError, match=r"rejected the torrent from https://nyaa.si/view/1 .*'Fails\.'"):
        service.add(
            item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=None), preview=False
        )


def test_add_unparseable_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the tracker parser yields no download URL, the add raises before qbit."""

    def _none_animetosho(url: str, client: httpx.Client) -> tuple[str | None, str]:
        del url, client
        return (None, "Title")

    monkeypatch.setattr(torrents, "get_animetosho_torrent", _none_animetosho)
    qbit = _FakeQbit()
    service = _service(qbit)

    with pytest.raises(TorrentParseError, match="Could not extract a torrent download link from"):
        service.add(
            item=SeadexUrlItem(url="https://animetosho.org/view/1", tracker=Tracker.ANIMETOSHO, infohash=None),
            preview=False,
        )

    assert qbit.add_calls == []


def _patch_all_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub all three tracker parsers to a fixed (url, title), so the parseable branch of `add` reaches the qbit path without a real scrape."""

    def _nyaa(url: str, client: httpx.Client) -> tuple[str | None, str]:
        del url, client
        return (_PARSED_URL, _SOURCE_TITLE)

    def _animetosho(url: str, client: httpx.Client) -> tuple[str | None, str]:
        del url, client
        return (_PARSED_URL, _SOURCE_TITLE)

    def _rutracker(url: str, infohash: str | None, client: httpx.Client) -> tuple[str | None, str]:
        del url, infohash, client
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _nyaa)
    monkeypatch.setattr(torrents, "get_animetosho_torrent", _animetosho)
    monkeypatch.setattr(torrents, "get_rutracker_torrent", _rutracker)


@pytest.mark.parametrize("tracker", list(Tracker))
def test_add_raises_iff_tracker_unparseable(tracker: Tracker, monkeypatch: pytest.MonkeyPatch) -> None:
    """`add` parses exactly the `PARSEABLE_TRACKERS` and raises for every other member.

    The constant derives from the parser table, so this pins the raise
    behavior rather than a lockstep between two hand-maintained collections.
    """

    _patch_all_parsers(monkeypatch)
    service = _service(_FakeQbit())

    if tracker in PARSEABLE_TRACKERS:
        outcome, _, _ = service.add(
            item=SeadexUrlItem(url="https://example/1", tracker=tracker, infohash=None), preview=True
        )
        assert outcome is AddOutcome.ADDED
    else:
        with pytest.raises(ValueError, match="Unable to parse torrent links"):
            service.add(item=SeadexUrlItem(url="https://example/1", tracker=tracker, infohash=None), preview=True)


def test_grab_failures_covers_every_expected_boundary_error() -> None:
    """Drift guard: the containment tuple catches each boundary's failure mode.

    The modes are the parse/add raises, an httpx blip (the tracker scrapes and
    the seadex transport all ride httpx), and any qbittorrentapi error.
    """

    expected = (
        TorrentParseError("x"),
        TorrentAddError("x"),
        httpx.ConnectError("x"),
        qbittorrentapi.APIConnectionError("x"),
    )
    for exc in expected:
        assert isinstance(exc, GRAB_FAILURES), f"{type(exc).__name__} not contained"
    # A ValueError (the unhandled-tracker defensive raise) must NOT be contained.
    assert not isinstance(ValueError("x"), GRAB_FAILURES)


# --- QbitConnectionError mapping (RunDeps.build login path) ------------------


def test_qbit_login_failure_maps_to_qbit_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A qBittorrent `APIConnectionError` at login is remapped to the operator-facing `QbitConnectionError` by `RunDeps.build`."""

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
            logger=logging.getLogger("pearlarr.test"),
            mappings=make_bare_instance(MappingResolver),
            web=httpx.Client(),
            boot=BootFlow(),
        )


@pytest.mark.parametrize(
    ("added_on", "expected"),
    [
        pytest.param(1_726_000_000, datetime.fromtimestamp(1_726_000_000), id="valid-epoch"),
        pytest.param(1e30, None, id="overflow"),
        pytest.param(True, None, id="bool"),
        pytest.param(0, None, id="zero"),
        pytest.param("2026-01-01", None, id="string"),
    ],
)
def test_add_already_present_reads_the_add_time(
    monkeypatch: pytest.MonkeyPatch, added_on: object, expected: datetime | None
) -> None:
    """A reacquire carries qBittorrent's add time through, folding junk readings to None."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit(present={_HASH: "Existing Name"}, added_on=added_on)
    service = _service(qbit)

    outcome, _, when = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH), preview=False
    )

    assert outcome is AddOutcome.ALREADY_ADDED
    assert when == expected


def test_add_preview_with_the_torrent_in_qbit_reports_it_without_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run checks qBittorrent first too: a torrent it has is ALREADY_ADDED and its page is never fetched."""

    fetched: list[str] = []

    def _recording_nyaa(url: str, client: httpx.Client) -> tuple[str, str]:
        del client
        fetched.append(url)
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _recording_nyaa)
    qbit = _FakeQbit(present={_HASH: "Existing Name"})
    service = _service(qbit)

    result = service.add(
        item=SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH), preview=True
    )

    assert (result.outcome, result.name) == (AddOutcome.ALREADY_ADDED, "Existing Name")
    assert fetched == []
    assert qbit.add_calls == []


def test_add_a_torrent_two_entries_list_is_fetched_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second entry listing a torrent finds it in qBittorrent, so its page is fetched once in all."""

    fetched: list[str] = []

    def _recording_nyaa(url: str, client: httpx.Client) -> tuple[str, str]:
        del client
        fetched.append(url)
        return (_PARSED_URL, _SOURCE_TITLE)

    monkeypatch.setattr(torrents, "get_nyaa_torrent", _recording_nyaa)
    qbit = _FakeQbit(register_on_add=(_HASH, "Qbit Reported Name"))
    service = _service(qbit)
    item = SeadexUrlItem(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash=_HASH)

    first = service.add(item=item, preview=False)
    second = service.add(item=item, preview=False)

    assert (first.outcome, second.outcome) == (AddOutcome.ADDED, AddOutcome.ALREADY_ADDED)
    assert fetched == ["https://nyaa.si/view/1"]


# --- TorrentService.mark_cross_seeds and a marked url's add -------------------

_SEADEX = "b" * 40
_CROSS = "c" * 40
_NEWER = "d" * 40
_EPISODES = [_TorrFile("Show/Show - 01.mkv", 100), _TorrFile("Show/Show - 02.mkv", 200)]
"""The url's files as qBittorrent lists them inside a torrent's folder."""
_HELD: Mapping[str, list[_TorrFile] | Exception] = {_CROSS: _EPISODES}


def _url() -> SeadexUrlItem:
    """SeaDex's public url for `_EPISODES`, under its own hash."""

    return url_item(files=["Show - 01.mkv", "Show - 02.mkv"], size=[100, 200], tracker=Tracker.NYAA, infohash=_SEADEX)


def _row(infohash: str, *, category: str = _GRAB, state: str = "downloading", progress: float = 0.5) -> _TorrInfo:
    """A torrent at `_EPISODES`'s total size, downloading and half done unless told otherwise."""

    return _TorrInfo(
        infohash, "Cross", added_on=1_700_000_000, category=category, total_size=300, state=state, progress=progress
    )


def _marked(service: TorrentService, item: SeadexUrlItem, *, pearlarr_imports: bool = True) -> str | None:
    """Mark one title's dict holding `item` on a run Pearlarr imports for, and return the hash it was marked with."""

    service.begin_run(pearlarr_imports=pearlarr_imports)
    service.mark_cross_seeds({"Group": SeadexReleaseGroupItem(urls={item.url: item})})
    return item.cross_seed_hash


@pytest.mark.parametrize(
    ("row", "pearlarr_imports"),
    [
        pytest.param(_row(_CROSS), True, id="downloading"),
        pytest.param(_row(_CROSS), False, id="downloading-arr-imports"),
        pytest.param(_row(_CROSS, category=_POST, state="stalledUP", progress=1.0), True, id="seeding-after-import"),
    ],
)
def test_mark_cross_seeds_marks_a_torrent_holding_the_urls_files(row: _TorrInfo, pearlarr_imports: bool) -> None:
    """A counted torrent holding exactly the url's files under another hash marks the url with that hash."""

    qbit = _FakeQbit(rows=(row,), files=_HELD)

    assert _marked(_service(qbit), _url(), pearlarr_imports=pearlarr_imports) == _CROSS


@pytest.mark.parametrize(
    ("rows", "files", "pearlarr_imports"),
    [
        pytest.param(
            (_row(_CROSS),), {_CROSS: [_TorrFile("Show/Show - 01v2.mkv", 100), _EPISODES[1]]}, True, id="a-name-differs"
        ),
        pytest.param(
            (_row(_CROSS),),
            {_CROSS: [_EPISODES[0], replace(_EPISODES[1], priority=0)]},
            True,
            id="a-file-not-downloaded",
        ),
        pytest.param((_row(_CROSS, category="tv"),), _HELD, True, id="in-another-category"),
        pytest.param((_row(_CROSS, category=_POST),), _HELD, False, id="after-import-arr-imports"),
        pytest.param((_row(_CROSS), _row(_SEADEX, category="tv")), _HELD, True, id="seadexs-own-in-another-category"),
        pytest.param((_row(_CROSS, state="stalledUP", progress=1.0),), _HELD, False, id="finished-arr-imports"),
        pytest.param((_row(_CROSS, state="missingFiles"),), _HELD, True, id="errored"),
    ],
)
def test_mark_cross_seeds_leaves_a_url_no_counted_torrent_holds(
    rows: tuple[_TorrInfo, ...], files: Mapping[str, list[_TorrFile] | Exception], pearlarr_imports: bool
) -> None:
    """A url stays SeaDex's when no counted torrent holds exactly its files, or SeaDex's own is in qBittorrent."""

    qbit = _FakeQbit(rows=rows, files=files)

    assert _marked(_service(qbit), _url(), pearlarr_imports=pearlarr_imports) is None


@pytest.mark.parametrize(
    ("item", "rows"),
    [
        pytest.param(_url(), (_row(_CROSS), _row(_SEADEX)), id="seadexs-own-at-that-size"),
        pytest.param(replace(_url(), infohash=None), (_row(_CROSS),), id="no-hash"),
    ],
)
def test_mark_cross_seeds_skips_a_url_without_reading_a_file_list(
    item: SeadexUrlItem, rows: tuple[_TorrInfo, ...]
) -> None:
    """A url SeaDex's own torrent holds, or one with no hash (every private url), is left before any file-list read."""

    qbit = _FakeQbit(rows=rows, files=_HELD)

    assert _marked(_service(qbit), item) is None
    assert qbit.files_reads == []


def test_mark_cross_seeds_lists_once_a_run_and_reads_a_file_list_once() -> None:
    """The categories are listed at the run's first title, and only a torrent at a url's total size has its files read."""

    other_size = replace(_row(_NEWER), total_size=999)
    qbit = _FakeQbit(rows=(_row(_CROSS), other_size), files=_HELD)
    service = _service(qbit)

    marks = [_marked(service, _url()) for _ in range(2)]

    assert marks == [_CROSS, _CROSS]
    assert qbit.category_lists == [_GRAB, _POST]
    assert qbit.files_reads == [_CROSS]


@respx.mock
def test_mark_cross_seeds_looks_again_after_a_title_whose_grab_category_could_not_be_read() -> None:
    """A title checked while the arr's download clients can't be read is left unmarked, and the next title looks."""

    respx.get("http://arr.test/api/v3/downloadclient").side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=[download_client_json(sonarr_client_fields(grab=_GRAB, post_import=_POST))]),
    ]
    qbit = _FakeQbit(rows=(_row(_CROSS),), files=_HELD)
    sources = CategorySources(replace(bind_arr_http()[0], retries=0), cast("qbittorrentapi.Client", qbit))
    service = _service(qbit, categories=ArrCategoryResolver(Arr.SONARR, ArrSettings(), sources))

    assert [_marked(service, _url()) for _ in range(2)] == [None, _CROSS]


def test_mark_cross_seeds_takes_the_oldest_holder() -> None:
    """Of two torrents holding the url's files, the oldest marks it, though the newer one has finished."""

    newer = replace(_row(_NEWER, state="stalledUP", progress=1.0), added_on=1_700_000_500)
    qbit = _FakeQbit(rows=(newer, _row(_CROSS)), files={_CROSS: _EPISODES, _NEWER: _EPISODES})

    assert _marked(_service(qbit), _url()) == _CROSS


def test_mark_cross_seeds_passes_over_a_holder_gone_at_its_file_list() -> None:
    """A torrent that left qBittorrent before its files were read gives way to the next one."""

    newer = replace(_row(_NEWER), added_on=1_700_000_500)
    qbit = _FakeQbit(rows=(_row(_CROSS), newer), files={_NEWER: _EPISODES})

    assert _marked(_service(qbit), _url()) == _NEWER


@pytest.mark.parametrize(
    ("list_error", "files"),
    [
        pytest.param(qbittorrentapi.APIConnectionError("Connection refused"), _HELD, id="list"),
        pytest.param(None, {_CROSS: qbittorrentapi.APIConnectionError("Connection refused")}, id="file-list"),
    ],
)
def test_mark_cross_seeds_a_failed_read_warns_once_and_reads_nothing_more(
    list_error: Exception | None, files: Mapping[str, list[_TorrFile] | Exception]
) -> None:
    """A failed qBittorrent read warns once: the run's later titles go unmarked and read nothing again."""

    recording = install_recording_hub()
    qbit = _FakeQbit(rows=(_row(_CROSS),), files=files, list_error=list_error)
    service = _service(qbit)

    marks = [_marked(service, _url()) for _ in range(2)]

    assert marks == [None, None]
    warning = (
        "Could not read qBittorrent's torrents (APIConnectionError) - "
        "this run may add a release that another torrent already holds"
    )
    assert diagnostic_messages(recording, Severity.WARNING) == [warning]


@pytest.mark.parametrize(
    ("arr", "wait_mode", "marked"),
    [
        pytest.param(Arr.SONARR, ImportWaitMode.BLOCKING, _CROSS, id="pearlarr-imports"),
        pytest.param(Arr.SONARR, ImportWaitMode.OFF, None, id="wait-off"),
        pytest.param(Arr.RADARR, ImportWaitMode.BLOCKING, None, id="radarr"),
    ],
)
def test_a_runs_seadex_dict_carries_a_finished_cross_seed_only_when_pearlarr_imports(
    arr: Arr, wait_mode: ImportWaitMode, marked: str | None
) -> None:
    """The dict a run builds from a SeaDex entry is marked, a finished torrent counting only on a Sonarr run that waits."""

    qbit = _FakeQbit(rows=(_row(_CROSS, category=_POST, state="stalledUP", progress=1.0),), files=_HELD)
    services = RunServices(replace(make_run_deps(), torrents=_service(qbit)), arr)
    services.begin_run(RunContext(arr=arr, import_wait_mode=wait_mode))
    listed = make_torrent_record(
        release_group="Group", infohash=_SEADEX, file_names=("Show - 01.mkv", "Show - 02.mkv"), file_sizes=(100, 200)
    )

    seadex_dict = services.get_seadex_dict(make_entry_record(torrents=(listed,)))

    (item,) = seadex_dict["Group"].urls.values()
    assert item.cross_seed_hash == marked


def test_add_a_marked_url_is_its_cross_seed() -> None:
    """A marked url is ALREADY_ADDED as the torrent holding its files, and SeaDex's is never added."""

    qbit = _FakeQbit(present={_CROSS: "Cross"})
    item = replace(_url(), cross_seed_hash=_CROSS)

    result = _service(qbit).add(item=item, preview=False)

    assert (result.outcome, result.name) == (AddOutcome.ALREADY_ADDED, "Cross")
    assert qbit.add_calls == []


def test_add_a_marked_url_whose_cross_seed_left_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A marked url whose torrent left qBittorrent fails its grab instead of adding SeaDex's under its record."""

    _patch_nyaa_parser(monkeypatch)
    qbit = _FakeQbit()

    with pytest.raises(TorrentAddError, match="left qBittorrent"):
        _service(qbit).add(item=replace(_url(), cross_seed_hash=_CROSS), preview=False)

    assert qbit.add_calls == []
