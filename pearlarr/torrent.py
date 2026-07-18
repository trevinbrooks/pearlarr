"""Per-tracker parsers: turn a tracker page/feed URL into a download link and release title."""

from typing import NamedTuple
from urllib.parse import urlencode, urljoin

import httpx
import pynyaa
from bs4 import BeautifulSoup

from .json_narrow import is_json_list, is_json_obj
from .web_client import get_with_retries

ANIMETOSHO_FEED_URL = "https://animetosho.org/feed/json"
RUTRACKER_MAGNET_ANNOUNCE = "http://bt2.t-ru.org/ann?magnet"

# (connect, read) timeout for the tracker page/feed scrapes, so a hung tracker
# surfaces as a transient miss instead of blocking the run. The shared web
# client bakes the same bounds in, so only the pynyaa client below reads this.
TRACKER_REQUEST_TIMEOUT_S = (5, 30)


class TorrentParseError(Exception):
    """A tracker page/feed didn't yield the release's torrent link or title."""


class ParsedTorrent(NamedTuple):
    """A parsed tracker result: the download/magnet link and the release title."""

    link: str | None
    """The .torrent download link or magnet URI, or None when the source yields
    no matching link."""
    title: str


# pynyaa rides httpx: give its client the same bounds (and keep pynyaa's own
# User-Agent, which its default client would otherwise set).
_NYAA_SESSION = pynyaa.Nyaa(
    client=httpx.Client(
        headers={"User-Agent": f"pynyaa/{pynyaa.__version__} (https://pypi.org/project/pynyaa/)"},
        timeout=httpx.Timeout(TRACKER_REQUEST_TIMEOUT_S[1], connect=TRACKER_REQUEST_TIMEOUT_S[0]),
    ),
)


def get_nyaa_torrent(url: str) -> ParsedTorrent:
    """Get the Nyaa download link and release title from a Nyaa URL."""

    release = _NYAA_SESSION.get(url)

    return ParsedTorrent(release.torrent.url, release.title)


def get_animetosho_torrent(
    url: str,
    client: httpx.Client,
) -> ParsedTorrent:
    """Get the AnimeTosho download link and release title from a URL.

    Args:
        url: URL of the AnimeTosho release page
        client: Client to reuse for the two requests this makes to the
            same host.

    Returns:
        The .torrent download link (None if no matching link is found in the
        feed) and the human-readable release title scraped from the page.
    """

    # Start by getting the webpage, so we can get a title. A 5xx/Cloudflare page
    # would otherwise scrape as a misleading "no title" parse error.
    r = get_with_retries(client, url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    titles = soup.find_all("h2", attrs={"id": "title"})

    if len(titles) == 0:
        raise TorrentParseError(f"Could not find the torrent title on {url}")

    if len(titles) > 1:
        raise TorrentParseError(f"Found more than one torrent title on {url}")

    title = titles[0].text

    # Query the feed API for the matching release (encode the title so reserved
    # characters in it don't malform the query string)
    query_url = urljoin(ANIMETOSHO_FEED_URL, "?" + urlencode({"t": "search", "q": title}))
    r = get_with_retries(client, query_url)
    r.raise_for_status()
    try:
        j = r.json()
    except ValueError as e:
        # An HTML error body (e.g. an interstitial) isn't JSON: a parse miss.
        raise TorrentParseError(f"AnimeTosho feed returned a non-JSON response from {query_url}") from e

    # A JSON error object (rate limit / interstitial) instead of the expected
    # feed array would otherwise iterate as its string keys and crash on .get.
    if not is_json_list(j):
        raise TorrentParseError(f"AnimeTosho feed returned unexpected JSON (not a list) from {query_url}")

    # Find the feed entry whose link matches the page URL, skipping non-object
    # entries. A non-str torrent_url folds to None (no link found).
    parsed_url: str | None = None
    for entry in j:
        if not is_json_obj(entry):
            continue
        if entry.get("link") == url:
            raw_url = entry.get("torrent_url")
            parsed_url = raw_url if isinstance(raw_url, str) else None
            break

    return ParsedTorrent(parsed_url, title)


def get_rutracker_torrent(
    url: str,
    infohash: str | None,
    client: httpx.Client,
) -> ParsedTorrent:
    """Get the RuTracker magnet link and torrent title from a URL.

    Args:
        url: URL of the RuTracker topic
        infohash: The hash the magnet's `urn:btih` payload is built from.
        client: Client to reuse for the page fetch.

    Returns:
        The magnet link and the human-readable torrent title scraped from
        the page.

    Raises:
        TorrentParseError: If `infohash` is None (a magnet needs the hash) or
            the page carries no title.
    """

    # No hash means no valid magnet ("urn:btih:None" is garbage). Fail as the
    # usual parse miss before fetching anything.
    if infohash is None:
        raise TorrentParseError("RuTracker release has no infohash to build a magnet link from")

    # Pull the torrent title from souping the URL. Use the stdlib html.parser
    # (as the AnimeTosho scraper does) - lxml is not a dependency, and the page
    # only needs a single class lookup, so the built-in parser is plenty.
    r = get_with_retries(client, url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    main_title = soup.find("h1", attrs={"class": "maintitle"})
    if main_title is None:
        raise TorrentParseError(f"Could not find the torrent title on {url}")
    torrent_title = main_title.text

    params = {
        "xt": f"urn:btih:{infohash}",
        "tr": RUTRACKER_MAGNET_ANNOUNCE,
        "dn": torrent_title,
    }
    url_encoded = urlencode(params)
    parsed_url = f"magnet:?{url_encoded}"

    return ParsedTorrent(parsed_url, torrent_title)
