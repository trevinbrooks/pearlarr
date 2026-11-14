"""Per-tracker parsers: turn a tracker page/feed URL into a torrent link and torrent name."""

import re
from typing import NamedTuple
from urllib.parse import unquote, urlencode, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from .json_narrow import is_json_list, is_json_obj
from .web_client import get_with_retries, head_with_retries

ANIMETOSHO_FEED_URL = "https://animetosho.org/feed/json"
RUTRACKER_MAGNET_ANNOUNCE = "http://bt2.t-ru.org/ann?magnet"

_NYAA_VIEW_PATH = re.compile(r"/view/(\d+)/?")
# RFC 6266 makes parameter names and the charset tag case-insensitive.
_DISPOSITION_NAME_UTF8 = re.compile(r"filename\*\s*=\s*utf-8''([^;]*)", re.IGNORECASE)
_DISPOSITION_NAME_PLAIN = re.compile(r'(?<![\w*])filename\s*=\s*("[^"]*"|[^;]*)', re.IGNORECASE)


class TrackerPageError(Exception):
    """A tracker page/feed didn't yield the release's torrent link or torrent name."""


class TrackerLink(NamedTuple):
    """A parsed tracker result: the .torrent/magnet link and the torrent name."""

    link: str | None
    """The .torrent link or magnet URI, or None when the source yields
    no matching link."""
    title: str


def fetch_nyaa_link(url: str, client: httpx.Client) -> TrackerLink:
    """Get the Nyaa .torrent link and torrent name from a Nyaa release URL.

    The link is built from the Nyaa id in the URL, the way Nyaa serves every .torrent. One HEAD request on it checks
    the file is there and reads the torrent name from its filename, so neither the page nor the file is fetched.
    """

    match = _NYAA_VIEW_PATH.fullmatch(urlsplit(url).path)
    if match is None:
        raise TrackerPageError(f"Could not read a Nyaa release number from {url}")
    link = urljoin(url, f"/download/{match[1]}.torrent")
    r = head_with_retries(client, link)
    r.raise_for_status()
    # Nyaa percent-encodes both forms of the name. The UTF-8 `filename*` wins unless it is missing or empty.
    disposition = r.headers.get("content-disposition", "")
    forms = (_DISPOSITION_NAME_UTF8, _DISPOSITION_NAME_PLAIN)
    names = (m[1].strip().strip('"') for pattern in forms if (m := pattern.search(disposition)))
    encoded = next(filter(None, names), "")
    if not encoded:
        raise TrackerPageError(f"Could not read the torrent name from {link}")
    return TrackerLink(link, unquote(encoded).removesuffix(".torrent"))


def fetch_animetosho_link(
    url: str,
    client: httpx.Client,
) -> TrackerLink:
    """Get the AnimeTosho .torrent link and torrent name from a URL.

    Args:
        url: URL of the AnimeTosho release page
        client: Client to reuse for the two requests this makes to the
            same host.

    Returns:
        The .torrent link (None if no matching link is found in the
        feed) and the human-readable torrent name scraped from the page.
    """

    # Start by getting the webpage, so we can get the torrent name. A 5xx/Cloudflare
    # page would otherwise scrape as a misleading "no title" page error.
    r = get_with_retries(client, url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    titles = soup.find_all("h2", attrs={"id": "title"})

    if len(titles) == 0:
        raise TrackerPageError(f"Could not find the torrent name on {url}")

    if len(titles) > 1:
        raise TrackerPageError(f"Found more than one torrent name on {url}")

    title = titles[0].text

    # Query the feed API for the matching release (encode the name so reserved
    # characters in it don't malform the query string)
    query_url = urljoin(ANIMETOSHO_FEED_URL, "?" + urlencode({"t": "search", "q": title}))
    r = get_with_retries(client, query_url)
    r.raise_for_status()
    try:
        j = r.json()
    except ValueError as e:
        # An HTML error body (e.g. an interstitial) isn't JSON: a `TrackerPageError`.
        raise TrackerPageError(f"AnimeTosho feed returned a non-JSON response from {query_url}") from e

    # A JSON error object (rate limit / interstitial) instead of the expected
    # feed array would otherwise iterate as its string keys and crash on .get.
    if not is_json_list(j):
        raise TrackerPageError(f"AnimeTosho feed returned unexpected JSON (not a list) from {query_url}")

    # Find the feed element whose link matches the page URL, skipping non-object
    # elements. A non-str torrent_url converts to None (no link found).
    parsed_url: str | None = None
    for entry in j:
        if not is_json_obj(entry):
            continue
        if entry.get("link") == url:
            raw_url = entry.get("torrent_url")
            parsed_url = raw_url if isinstance(raw_url, str) else None
            break

    return TrackerLink(parsed_url, title)


def fetch_rutracker_link(
    url: str,
    infohash: str | None,
    client: httpx.Client,
) -> TrackerLink:
    """Get the RuTracker magnet link and torrent name from a URL.

    Args:
        url: URL of the RuTracker topic
        infohash: The hash the magnet's `urn:btih` payload is built from.
        client: Client to reuse for the page fetch.

    Returns:
        The magnet link and the human-readable torrent name scraped from
        the page.

    Raises:
        TrackerPageError: If `infohash` is None (a magnet needs the hash) or
            the page carries no torrent name.
    """

    # No hash means no valid magnet ("urn:btih:None" is garbage). Fail as the
    # usual `TrackerPageError` before fetching anything.
    if infohash is None:
        raise TrackerPageError("RuTracker release has no infohash to build a magnet link from")

    # Pull the torrent name from souping the URL. Use the stdlib html.parser
    # (as the AnimeTosho scraper does) - lxml is not a dependency, and the page
    # only needs a single class lookup, so the built-in parser is plenty.
    r = get_with_retries(client, url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    main_title = soup.find("h1", attrs={"class": "maintitle"})
    if main_title is None:
        raise TrackerPageError(f"Could not find the torrent name on {url}")
    torrent_title = main_title.text

    params = {
        "xt": f"urn:btih:{infohash}",
        "tr": RUTRACKER_MAGNET_ANNOUNCE,
        "dn": torrent_title,
    }
    url_encoded = urlencode(params)
    parsed_url = f"magnet:?{url_encoded}"

    return TrackerLink(parsed_url, torrent_title)
