from typing import Any, cast
from urllib.parse import urlencode, urljoin

import httpx
import pynyaa
from bs4 import BeautifulSoup

from .web_client import get_with_retries, make_web_client

ANIMETOSHO_FEED_URL = "https://animetosho.org/feed/json"
RUTRACKER_MAGNET_ANNOUNCE = "http://bt2.t-ru.org/ann?magnet"

# (connect, read) timeout for the tracker page/feed scrapes, so a hung tracker
# surfaces as a transient miss instead of blocking the run. The shared web
# client bakes the same bounds in, so only the pynyaa client below reads this.
TRACKER_REQUEST_TIMEOUT_S = (5, 30)


class TorrentParseError(Exception):
    """A tracker page/feed didn't yield the release's torrent link or title."""


# Reused when a caller doesn't pass its own client, so even standalone use of
# these helpers gets keep-alive pooling and the transient-retry policy. The
# main code path threads in RunDeps.web instead.
_DEFAULT_CLIENT = make_web_client()
# pynyaa rides httpx: give its client the same bounds (and keep pynyaa's own
# User-Agent, which its default client would otherwise set).
_NYAA_SESSION = pynyaa.Nyaa(
    client=httpx.Client(
        headers={"User-Agent": f"pynyaa/{pynyaa.__version__} (https://pypi.org/project/pynyaa/)"},
        timeout=httpx.Timeout(TRACKER_REQUEST_TIMEOUT_S[1], connect=TRACKER_REQUEST_TIMEOUT_S[0]),
    ),
)


def get_nyaa_torrent(url: str) -> tuple[str, str]:
    """Get the Nyaa download link and release title from a Nyaa URL

    Args:
        url (str): URL of the Nyaa release page

    Returns:
        tuple: (download_url, release_title) - the .torrent download link and
            the human-readable release title
    """

    release = _NYAA_SESSION.get(url)

    return release.torrent.url, release.title


def get_animetosho_torrent(
    url: str,
    client: httpx.Client | None = None,
) -> tuple[str | None, str]:
    """Get the AnimeTosho download link and release title from a URL

    Args:
        url (str): URL of the AnimeTosho release page
        client (httpx.Client, optional): Client to reuse for the two
            requests this makes to the same host. Defaults to a shared one.

    Returns:
        tuple: (download_url, release_title) - the .torrent download link
            (None if no matching link is found in the feed) and the
            human-readable release title scraped from the page
    """

    client = client or _DEFAULT_CLIENT

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
    if not isinstance(j, list):
        raise TorrentParseError(f"AnimeTosho feed returned unexpected JSON (not a list) from {query_url}")

    # Find the feed entry whose link matches the page URL. response.json() is
    # untyped, so cast each entry at the parse boundary (skipping non-objects).
    parsed_url: str | None = None
    for entry in cast("list[Any]", j):
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, Any]", entry)
        if item.get("link") == url:
            parsed_url = cast("str | None", item.get("torrent_url"))
            break

    return parsed_url, title


def get_rutracker_torrent(
    url: str,
    infohash: str | None,
    client: httpx.Client | None = None,
) -> tuple[str, str]:
    """Get the RuTracker magnet link and torrent title from a URL

    Args:
        url (str): URL of the RuTracker topic
        infohash (str | None): Torrent info hash
        client (httpx.Client, optional): Client to reuse for the page
            fetch. Defaults to a shared one.

    Returns:
        tuple: (magnet_url, torrent_title) - the magnet link and the
            human-readable torrent title scraped from the page

    Raises:
        TorrentParseError: If ``infohash`` is None (a magnet needs the hash) or
            the page carries no title.
    """

    # No hash means no valid magnet ("urn:btih:None" is garbage); fail as the
    # usual parse miss before fetching anything.
    if infohash is None:
        raise TorrentParseError("RuTracker release has no infohash to build a magnet link from")

    client = client or _DEFAULT_CLIENT

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

    return parsed_url, torrent_title
