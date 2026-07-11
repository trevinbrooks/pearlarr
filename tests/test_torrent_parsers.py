# pyright: strict
"""Direct tests for the tracker HTML/feed parsers in `torrent`.

The three `get_*_torrent` helpers scrape a release page (and, for AnimeTosho,
a JSON feed) into `(download_url, title)`. AnimeTosho and RuTracker take an
`httpx.Client`, so they are driven with saved HTML fixtures over a
`respx`-mocked boundary; Nyaa uses `pynyaa` (its own httpx client), so its
module-level session is swapped for a typed stub. The documented error raises
are exercised alongside the success paths; the 5xx paths stub the retry
helper's backoff sleep so retry exhaustion doesn't wait for real.
"""

import re
from pathlib import Path

import httpx
import pytest
import respx

import pearlarr.modules.torrent as torrent
from pearlarr.modules.torrent import (
    ANIMETOSHO_FEED_URL,
    TorrentParseError,
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)
from pearlarr.modules.web_client import get_with_retries

_TORRENT_FIXTURES = Path(__file__).parent / "fixtures" / "torrent"

# The titles the saved HTML fixtures carry (so a test asserts the exact scrape).
_ANIMETOSHO_TITLE = "[Erai-raws] Cool Anime - 01 [1080p][Multiple Subtitle]"
_RUTRACKER_TITLE = "[BDRemux 1080p] Cool Anime / クールアニメ [RUS+JAP] (12/12)"


def _torrent_fixture(name: str) -> str:
    """Read a saved tracker-page HTML fixture by file name."""

    return (_TORRENT_FIXTURES / name).read_text(encoding="utf-8")


def _stub_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-point the parsers' retry helper at a no-sleep twin.

    A 5xx response is transient, so `get_with_retries` retries it with real
    backoff sleeps; the retry policy itself is pinned in `test_web_client`,
    so these tests only need the exhausted response, not the waits.
    """

    def _no_sleep_get(client: httpx.Client, url: str) -> httpx.Response:
        return get_with_retries(client, url, sleep=lambda _s: None)

    monkeypatch.setattr(torrent, "get_with_retries", _no_sleep_get)


# --- Nyaa (pynyaa session swapped for a typed stub) -------------------------


class _StubNyaaTorrent:
    """The `.torrent` sub-object pynyaa exposes (only `.url` is read)."""

    def __init__(self, url: str) -> None:
        self.url = url


class _StubNyaaRelease:
    """A pynyaa release: `.torrent.url` (download link) + `.title`."""

    def __init__(self, torrent_url: str, title: str) -> None:
        self.torrent = _StubNyaaTorrent(torrent_url)
        self.title = title


class _StubNyaa:
    """Stand-in for `pynyaa.Nyaa` recording the URL it was asked to fetch."""

    def __init__(self, release: _StubNyaaRelease) -> None:
        self._release = release
        self.calls: list[str] = []

    def get(self, url: str) -> _StubNyaaRelease:
        self.calls.append(url)
        return self._release


def test_get_nyaa_torrent_returns_download_and_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Nyaa parser returns `(release.torrent.url, release.title)` verbatim."""

    release = _StubNyaaRelease(
        torrent_url="https://nyaa.si/download/1.torrent",
        title="[Group] Cool Anime - 01 [1080p]",
    )
    stub = _StubNyaa(release)
    monkeypatch.setattr(torrent, "_NYAA_SESSION", stub)

    result = get_nyaa_torrent("https://nyaa.si/view/1")

    assert result == ("https://nyaa.si/download/1.torrent", "[Group] Cool Anime - 01 [1080p]")
    assert stub.calls == ["https://nyaa.si/view/1"]


# --- AnimeTosho (scrape the page title, then look it up in the JSON feed) ----


@respx.mock
def test_get_animetosho_torrent_success() -> None:
    """The scraped page title plus the feed entry whose `link` matches the URL
    yield `(torrent_url, title)`.
    """

    page_url = "https://animetosho.org/view/cool-anime-01.123456"
    feed_torrent = "https://animetosho.org/storage/torrent/abc/cool-anime-01.torrent"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(
        json=[
            {"link": "https://animetosho.org/view/some-other.999", "torrent_url": "https://other/x.torrent"},
            {"link": page_url, "torrent_url": feed_torrent},
        ],
    )
    result = get_animetosho_torrent(page_url, client=httpx.Client())

    assert result == (feed_torrent, _ANIMETOSHO_TITLE)


@respx.mock
def test_get_animetosho_torrent_no_feed_match_returns_none_url() -> None:
    """When no feed entry's `link` matches the page URL, the download URL is
    `None` but the scraped title is still returned.
    """

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(
        json=[{"link": "https://animetosho.org/view/unrelated.1", "torrent_url": "https://other/x.torrent"}],
    )
    download_url, title = get_animetosho_torrent(page_url, client=httpx.Client())

    assert download_url is None
    assert title == _ANIMETOSHO_TITLE


@respx.mock
def test_get_animetosho_torrent_non_str_url_folds_to_none() -> None:
    """A matching feed entry with a junk (non-str) `torrent_url` yields `None`
    instead of leaking the junk value downstream as a fake download URL.
    """

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(json=[{"link": page_url, "torrent_url": 123}])
    download_url, title = get_animetosho_torrent(page_url, client=httpx.Client())

    assert download_url is None
    assert title == _ANIMETOSHO_TITLE


@respx.mock
def test_get_animetosho_torrent_missing_title_raises() -> None:
    """A page with no `<h2 id="title">` raises before the feed is queried."""

    page_url = "https://animetosho.org/view/no-title.1"

    respx.get(page_url).respond(html="<html><body><p>no title here</p></body></html>")
    with pytest.raises(TorrentParseError, match=f"Could not find the torrent title on {re.escape(page_url)}"):
        get_animetosho_torrent(page_url, client=httpx.Client())


@respx.mock
def test_get_animetosho_torrent_two_titles_raises() -> None:
    """A page with more than one `<h2 id="title">` is ambiguous and raises."""

    page_url = "https://animetosho.org/view/two-titles.1"

    respx.get(page_url).respond(html='<html><body><h2 id="title">First</h2><h2 id="title">Second</h2></body></html>')
    with pytest.raises(TorrentParseError, match="more than one torrent title"):
        get_animetosho_torrent(page_url, client=httpx.Client())


@respx.mock
def test_get_animetosho_torrent_http_500_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx page (still 5xx once the transient retries run out) raises
    `HTTPStatusError` (a contained grab failure) instead of scraping the
    error body into a misleading "no title" parse error."""

    _stub_retry_sleep(monkeypatch)
    page_url = "https://animetosho.org/view/down.1"

    respx.get(page_url).respond(status_code=500, html="<html>Server Error</html>")
    with pytest.raises(httpx.HTTPStatusError):
        get_animetosho_torrent(page_url, client=httpx.Client())


@respx.mock
def test_get_animetosho_torrent_non_json_feed_is_a_parse_error() -> None:
    """An HTML error body from the feed (HTTP 200 but not JSON) surfaces as a
    `TorrentParseError` naming the feed URL, not a raw `JSONDecodeError`."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(html="<html>interstitial</html>")
    with pytest.raises(TorrentParseError, match="non-JSON response"):
        get_animetosho_torrent(page_url, client=httpx.Client())


@respx.mock
def test_get_animetosho_torrent_non_list_json_is_a_parse_error() -> None:
    """A JSON error OBJECT from the feed (rate limit / API error) surfaces as a
    `TorrentParseError`, not an AttributeError from iterating its keys."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(json={"error": "rate limited"})
    with pytest.raises(TorrentParseError, match="not a list"):
        get_animetosho_torrent(page_url, client=httpx.Client())


# --- RuTracker (scrape the maintitle, build the magnet locally) --------------


@respx.mock
def test_get_rutracker_torrent_builds_magnet() -> None:
    """The RuTracker parser scrapes the maintitle and builds the magnet from the
    hash, the fixed announce, and the title as `dn`.
    """

    url = "https://rutracker.org/forum/viewtopic.php?t=1234567"
    infohash = "abcdef0123456789abcdef0123456789abcdef01"

    respx.get(url).respond(html=_torrent_fixture("rutracker_page.html"))
    magnet, title = get_rutracker_torrent(url, infohash, client=httpx.Client())

    assert title == _RUTRACKER_TITLE
    assert magnet.startswith(f"magnet:?xt=urn%3Abtih%3A{infohash}")
    assert "tr=http" in magnet
    assert "dn=" in magnet


@respx.mock
def test_get_rutracker_torrent_missing_title_raises() -> None:
    """A page with no `h1.maintitle` raises."""

    url = "https://rutracker.org/forum/viewtopic.php?t=7654321"

    respx.get(url).respond(html="<html><body><div>no maintitle</div></body></html>")
    with pytest.raises(TorrentParseError, match=f"Could not find the torrent title on {re.escape(url)}"):
        get_rutracker_torrent(url, "deadbeef", client=httpx.Client())


@respx.mock
def test_get_rutracker_torrent_http_500_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx topic page raises `HTTPStatusError` rather than a misleading parse error."""

    _stub_retry_sleep(monkeypatch)
    url = "https://rutracker.org/forum/viewtopic.php?t=1234567"

    respx.get(url).respond(status_code=500, html="<html>Server Error</html>")
    with pytest.raises(httpx.HTTPStatusError):
        get_rutracker_torrent(url, "deadbeef", client=httpx.Client())


@respx.mock
def test_get_rutracker_torrent_no_infohash_raises() -> None:
    """A None infohash can't make a magnet ("urn:btih:None"): raise the parse
    error before any fetch (the routeless mock proves nothing was requested)."""

    with pytest.raises(TorrentParseError, match="no infohash"):
        get_rutracker_torrent(
            "https://rutracker.org/forum/viewtopic.php?t=1",
            None,
            client=httpx.Client(),
        )
