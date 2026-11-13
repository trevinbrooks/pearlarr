# pyright: strict
"""Direct tests for the tracker HTML/feed parsers in `tracker_links`.

The three `fetch_*_link` helpers turn a release URL into a `TrackerLink` (link, torrent name):
Nyaa from one HEAD on its .torrent, AnimeTosho from its page and JSON feed,
RuTracker from its page. Each takes an `httpx.Client`, so they are driven over a
`respx`-mocked boundary, the pages with saved HTML fixtures. The documented error raises
are exercised alongside the success paths. The 5xx paths stub the retry
helper's backoff sleep so retry exhaustion doesn't wait for real.
"""

import re
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
import respx

import pearlarr.tracker_links as torrent
from pearlarr.tracker_links import (
    ANIMETOSHO_FEED_URL,
    TrackerLink,
    TrackerPageError,
    fetch_animetosho_link,
    fetch_nyaa_link,
    fetch_rutracker_link,
)
from pearlarr.web_client import get_with_retries, head_with_retries

_TORRENT_FIXTURES = Path(__file__).parent / "fixtures" / "torrent"

# The torrent names the saved HTML fixtures carry (so a test asserts the exact scrape).
_NYAA_TITLE = "[Group] Cool Anime - 01 [1080p]"
_NYAA_LINK = "https://nyaa.si/download/1234567.torrent"
_ANIMETOSHO_TITLE = "[Erai-raws] Cool Anime - 01 [1080p][Multiple Subtitle]"
_RUTRACKER_TITLE = "[BDRemux 1080p] Cool Anime / クールアニメ [RUS+JAP] (12/12)"


def _torrent_fixture(name: str) -> str:
    """Read a saved tracker-page HTML fixture by file name."""

    return (_TORRENT_FIXTURES / name).read_text(encoding="utf-8")


def _stub_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-point the parsers' retry helper at a no-sleep twin.

    A 5xx response is transient, so `get_with_retries` retries it with real
    backoff sleeps. The retry policy itself is pinned in `test_web_client`,
    so these tests only need the exhausted response, not the waits.
    """

    def _no_sleep_get(client: httpx.Client, url: str) -> httpx.Response:
        return get_with_retries(client, url, sleep=lambda _s: None)

    monkeypatch.setattr(torrent, "get_with_retries", _no_sleep_get)


# --- Nyaa (build the .torrent link, read the torrent name off its download header) ---


def _nyaa_disposition(filename: str) -> dict[str, str]:
    """The download header Nyaa sends: the name twice, percent-encoded, the second as RFC 6266 `filename*`."""

    encoded = quote(filename)
    return {"content-disposition": f"inline; filename=\"{encoded}\"; filename*=UTF-8''{encoded}"}


@respx.mock
def test_fetch_nyaa_link_reads_the_title_off_one_head() -> None:
    """The link is built from the release number and the torrent name is its download filename, from one HEAD."""

    head = respx.head(_NYAA_LINK).respond(headers=_nyaa_disposition(f"{_NYAA_TITLE}.torrent"))

    result = fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client())

    assert result == (_NYAA_LINK, _NYAA_TITLE)
    # Any other request (the page, or a GET of the file) would fail the routeless mock.
    assert head.call_count == 1


@respx.mock
def test_fetch_nyaa_link_decodes_a_non_ascii_title() -> None:
    """The torrent name comes from the header's UTF-8 `filename*` form, so a non-ASCII name survives."""

    respx.head(_NYAA_LINK).respond(headers=_nyaa_disposition("[Group] クールアニメ - 01.torrent"))

    _, title = fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client())

    assert title == "[Group] クールアニメ - 01"


@pytest.mark.parametrize(
    "url", ["https://nyaa.si/view/1234567/", "https://nyaa.si/view/1234567#com", "https://nyaa.si/view/1234567?page=2"]
)
@respx.mock
def test_fetch_nyaa_link_finds_the_release_number_past_a_slash_fragment_or_query(url: str) -> None:
    respx.head(_NYAA_LINK).respond(headers=_nyaa_disposition(f"{_NYAA_TITLE}.torrent"))

    assert fetch_nyaa_link(url, client=httpx.Client()) == TrackerLink(_NYAA_LINK, _NYAA_TITLE)


@respx.mock
def test_fetch_nyaa_link_refuses_a_download_url_without_a_request() -> None:
    head = respx.head(_NYAA_LINK).respond(headers=_nyaa_disposition(f"{_NYAA_TITLE}.torrent"))

    with pytest.raises(TrackerPageError, match="Could not read a Nyaa release number"):
        fetch_nyaa_link(_NYAA_LINK, client=httpx.Client())

    assert not head.called


@respx.mock
def test_fetch_nyaa_link_reads_the_utf8_name_in_any_case() -> None:
    respx.head(_NYAA_LINK).respond(
        headers={"content-disposition": f"attachment; FILENAME*=utf-8''{_NYAA_TITLE}.torrent"}
    )

    assert fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client()).title == _NYAA_TITLE


@pytest.mark.parametrize(
    "template",
    [
        pytest.param('filename="{name}.torrent"', id="quoted"),
        pytest.param("filename={name}.torrent", id="bare"),
        pytest.param("filename=\"{name}.torrent\"; filename*=UTF-8''", id="empty utf-8 form"),
        pytest.param('FILENAME="{name}.torrent"', id="any case"),
        pytest.param('origfilename="other.torrent"; filename="{name}.torrent"', id="past a longer parameter name"),
    ],
)
@respx.mock
def test_fetch_nyaa_link_reads_the_plain_filename(template: str) -> None:
    value = template.format(name=quote(_NYAA_TITLE))
    respx.head(_NYAA_LINK).respond(headers={"content-disposition": f"inline; {value}"})

    assert fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client()).title == _NYAA_TITLE


@respx.mock
def test_fetch_nyaa_link_retries_a_429(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_sleep_head(client: httpx.Client, url: str) -> httpx.Response:
        return head_with_retries(client, url, sleep=lambda _s: None)

    monkeypatch.setattr(torrent, "head_with_retries", _no_sleep_head)
    route = respx.head(_NYAA_LINK)
    route.side_effect = [httpx.Response(429), httpx.Response(200, headers=_nyaa_disposition(f"{_NYAA_TITLE}.torrent"))]

    _, title = fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client())

    assert title == _NYAA_TITLE
    assert route.call_count == 2


@respx.mock
def test_fetch_nyaa_link_without_a_release_number_raises() -> None:
    """A URL with no `/view/<number>` raises before any fetch (the routeless mock proves nothing was requested)."""

    url = "https://nyaa.si/user/x"

    with pytest.raises(TrackerPageError, match=f"Could not read a Nyaa release number from {re.escape(url)}"):
        fetch_nyaa_link(url, client=httpx.Client())


@respx.mock
def test_fetch_nyaa_link_without_a_filename_raises() -> None:
    """A download header with no file name raises."""

    respx.head(_NYAA_LINK).respond(headers={"content-disposition": "inline"})
    with pytest.raises(TrackerPageError, match=f"Could not read the torrent name from {re.escape(_NYAA_LINK)}"):
        fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client())


@respx.mock
def test_fetch_nyaa_link_deleted_release_raises_http_error() -> None:
    """A 404 on the .torrent (a deleted release) raises `HTTPStatusError`, a contained grab failure."""

    respx.head(_NYAA_LINK).respond(status_code=404)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_nyaa_link("https://nyaa.si/view/1234567", client=httpx.Client())


# --- AnimeTosho (scrape the torrent name, then look it up in the JSON feed) ---


@respx.mock
def test_fetch_animetosho_link_success() -> None:
    """The scraped torrent name plus the feed element whose `link` matches the URL yield `(torrent_url, title)`."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"
    feed_torrent = "https://animetosho.org/storage/torrent/abc/cool-anime-01.torrent"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(
        json=[
            {"link": "https://animetosho.org/view/some-other.999", "torrent_url": "https://other/x.torrent"},
            {"link": page_url, "torrent_url": feed_torrent},
        ],
    )
    result = fetch_animetosho_link(page_url, client=httpx.Client())

    assert result == (feed_torrent, _ANIMETOSHO_TITLE)


@respx.mock
def test_fetch_animetosho_link_no_feed_match_returns_none_url() -> None:
    """When no feed element's `link` matches the page URL, the .torrent link is `None` but the scraped torrent name is still returned."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(
        json=[{"link": "https://animetosho.org/view/unrelated.1", "torrent_url": "https://other/x.torrent"}],
    )
    download_url, title = fetch_animetosho_link(page_url, client=httpx.Client())

    assert download_url is None
    assert title == _ANIMETOSHO_TITLE


@respx.mock
def test_fetch_animetosho_link_non_str_url_becomes_none() -> None:
    """A matching feed element with a junk (non-str) `torrent_url` yields `None` instead of leaking the junk value downstream as a fake .torrent link."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(json=[{"link": page_url, "torrent_url": 123}])
    download_url, title = fetch_animetosho_link(page_url, client=httpx.Client())

    assert download_url is None
    assert title == _ANIMETOSHO_TITLE


@respx.mock
def test_fetch_animetosho_link_missing_title_raises() -> None:
    """A page with no `<h2 id="title">` raises before the feed is queried."""

    page_url = "https://animetosho.org/view/no-title.1"

    respx.get(page_url).respond(html="<html><body><p>no title here</p></body></html>")
    with pytest.raises(TrackerPageError, match=f"Could not find the torrent title on {re.escape(page_url)}"):
        fetch_animetosho_link(page_url, client=httpx.Client())


@respx.mock
def test_fetch_animetosho_link_two_titles_raises() -> None:
    """A page with more than one `<h2 id="title">` is ambiguous and raises."""

    page_url = "https://animetosho.org/view/two-titles.1"

    respx.get(page_url).respond(html='<html><body><h2 id="title">First</h2><h2 id="title">Second</h2></body></html>')
    with pytest.raises(TrackerPageError, match="more than one torrent title"):
        fetch_animetosho_link(page_url, client=httpx.Client())


@respx.mock
def test_fetch_animetosho_link_http_500_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx page (still 5xx once the transient retries run out) raises `HTTPStatusError` (a contained grab failure) instead of scraping the error body into a misleading "no title" parse error."""

    _stub_retry_sleep(monkeypatch)
    page_url = "https://animetosho.org/view/down.1"

    respx.get(page_url).respond(status_code=500, html="<html>Server Error</html>")
    with pytest.raises(httpx.HTTPStatusError):
        fetch_animetosho_link(page_url, client=httpx.Client())


@respx.mock
def test_fetch_animetosho_link_non_json_feed_is_a_parse_error() -> None:
    """An HTML error body from the feed (HTTP 200 but not JSON) surfaces as a `TrackerPageError` naming the feed URL, not a raw `JSONDecodeError`."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(html="<html>interstitial</html>")
    with pytest.raises(TrackerPageError, match="non-JSON response"):
        fetch_animetosho_link(page_url, client=httpx.Client())


@respx.mock
def test_fetch_animetosho_link_non_list_json_is_a_parse_error() -> None:
    """A JSON error OBJECT from the feed (rate limit / API error) surfaces as a `TrackerPageError`, not an AttributeError from iterating its keys."""

    page_url = "https://animetosho.org/view/cool-anime-01.123456"

    respx.get(page_url).respond(html=_torrent_fixture("animetosho_page.html"))
    respx.get(ANIMETOSHO_FEED_URL).respond(json={"error": "rate limited"})
    with pytest.raises(TrackerPageError, match="not a list"):
        fetch_animetosho_link(page_url, client=httpx.Client())


# --- RuTracker (scrape the maintitle, build the magnet locally) --------------


@respx.mock
def test_fetch_rutracker_link_builds_magnet() -> None:
    """The RuTracker parser scrapes the maintitle and builds the magnet from the hash, the fixed announce, and the torrent name as `dn`."""

    url = "https://rutracker.org/forum/viewtopic.php?t=1234567"
    infohash = "abcdef0123456789abcdef0123456789abcdef01"

    respx.get(url).respond(html=_torrent_fixture("rutracker_page.html"))
    magnet, title = fetch_rutracker_link(url, infohash, client=httpx.Client())

    assert title == _RUTRACKER_TITLE
    # RuTracker always builds a magnet (never a None link). Narrow for the checks below.
    assert magnet is not None
    assert magnet.startswith(f"magnet:?xt=urn%3Abtih%3A{infohash}")
    assert "tr=http" in magnet
    assert "dn=" in magnet


@respx.mock
def test_fetch_rutracker_link_missing_title_raises() -> None:
    """A page with no `h1.maintitle` raises."""

    url = "https://rutracker.org/forum/viewtopic.php?t=7654321"

    respx.get(url).respond(html="<html><body><div>no maintitle</div></body></html>")
    with pytest.raises(TrackerPageError, match=f"Could not find the torrent title on {re.escape(url)}"):
        fetch_rutracker_link(url, "deadbeef", client=httpx.Client())


@respx.mock
def test_fetch_rutracker_link_http_500_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx topic page raises `HTTPStatusError` rather than a misleading parse error."""

    _stub_retry_sleep(monkeypatch)
    url = "https://rutracker.org/forum/viewtopic.php?t=1234567"

    respx.get(url).respond(status_code=500, html="<html>Server Error</html>")
    with pytest.raises(httpx.HTTPStatusError):
        fetch_rutracker_link(url, "deadbeef", client=httpx.Client())


@respx.mock
def test_fetch_rutracker_link_no_infohash_raises() -> None:
    """A None infohash can't make a magnet ("urn:btih:None"): raise the parse error before any fetch (the routeless mock proves nothing was requested)."""

    with pytest.raises(TrackerPageError, match="no infohash"):
        fetch_rutracker_link(
            "https://rutracker.org/forum/viewtopic.php?t=1",
            None,
            client=httpx.Client(),
        )
