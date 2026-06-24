from urllib.parse import urlencode, urljoin

import pynyaa
import requests
from bs4 import BeautifulSoup

ANIMETOSHO_FEED_URL = "https://animetosho.org/feed/json"
RUTRACKER_MAGNET_ANNOUNCE = "http://bt2.t-ru.org/ann?magnet"

# Reused when a caller doesn't pass its own session, so even standalone use of
# these helpers gets keep-alive connection pooling. The main code path threads
# in SeaDexArr.session instead.
_DEFAULT_SESSION = requests.Session()
_NYAA_SESSION = pynyaa.Nyaa()


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
    session: requests.Session | None = None,
) -> tuple[str | None, str]:
    """Get the AnimeTosho download link and release title from a URL

    Args:
        url (str): URL of the AnimeTosho release page
        session (requests.Session, optional): Session to reuse for the two
            requests this makes to the same host. Defaults to a shared one.

    Returns:
        tuple: (download_url, release_title) - the .torrent download link
            (None if no matching link is found in the feed) and the
            human-readable release title scraped from the page
    """

    session = session or _DEFAULT_SESSION

    # Start by getting the webpage, so we can get a title
    r = session.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    titles = soup.find_all("h2", attrs={"id": "title"})

    if len(titles) == 0:
        raise Exception("Could not find torrent name in AnimeTosho webpage")

    if len(titles) > 1:
        raise Exception("More than one torrent title in AnimeTosho webpage")

    title = titles[0].text

    # Query the feed API for the matching release (encode the title so reserved
    # characters in it don't malform the query string)
    query_url = urljoin(ANIMETOSHO_FEED_URL, "?" + urlencode({"t": "search", "q": title}))
    r = session.get(query_url)
    j = r.json()

    # Find the feed entry whose link matches the page URL
    parsed_url = None
    for i in j:
        if i.get("link", None) == url:
            parsed_url = i.get("torrent_url", None)
            break

    return parsed_url, title


def get_rutracker_torrent(
    url: str,
    torrent_hash: str | None,
    session: requests.Session | None = None,
) -> tuple[str, str]:
    """Get the RuTracker magnet link and torrent title from a URL

    Args:
        url (str): URL of the RuTracker topic
        torrent_hash (str | None): Torrent hash
        session (requests.Session, optional): Session to reuse for the page
            fetch. Defaults to a shared one.

    Returns:
        tuple: (magnet_url, torrent_title) - the magnet link and the
            human-readable torrent title scraped from the page
    """

    session = session or _DEFAULT_SESSION

    # Pull the torrent title from souping the URL. Use the stdlib html.parser
    # (as the AnimeTosho scraper does) - lxml is not a dependency, and the page
    # only needs a single class lookup, so the built-in parser is plenty.
    r = session.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    main_title = soup.find("h1", attrs={"class": "maintitle"})
    if main_title is None:
        raise Exception("Could not find torrent title in RuTracker webpage")
    torrent_title = main_title.text

    params = {
        "xt": f"urn:btih:{torrent_hash}",
        "tr": RUTRACKER_MAGNET_ANNOUNCE,
        "dn": torrent_title,
    }
    url_encoded = urlencode(params)
    parsed_url = f"magnet:?{url_encoded}"

    return parsed_url, torrent_title
