from urllib.parse import urlencode, urljoin

import pynyaa
import requests
from bs4 import BeautifulSoup

ANIMETOSHO_FEED_URL = "https://animetosho.org/feed/json"
RUTRACKER_MAGNET_ANNOUNCE = "http://bt2.t-ru.org/ann?magnet"


def get_nyaa_torrent(url: str) -> tuple[str, str]:
    """Get the Nyaa download link and release title from a Nyaa URL

    Args:
        url (str): URL of the Nyaa release page

    Returns:
        tuple: (download_url, release_title) - the .torrent download link and
            the human-readable release title
    """

    release = pynyaa.get(url)

    return release.torrent.url, release.title


def get_animetosho_torrent(url: str) -> tuple[str | None, str]:
    """Get the AnimeTosho download link and release title from a URL

    Args:
        url (str): URL of the AnimeTosho release page

    Returns:
        tuple: (download_url, release_title) - the .torrent download link
            (None if no matching link is found in the feed) and the
            human-readable release title scraped from the page
    """

    # Start by getting the webpage, so we can get a title
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    titles = soup.find_all("h2", attrs={"id": "title"})

    if len(titles) == 0:
        raise Exception("Could not find torrent name in AnimeTosho webpage")

    if len(titles) > 1:
        raise Exception("More than one torrent title in AnimeTosho webpage")

    title = titles[0].text

    # Fantastic, we have a title. Now query API
    query_url = urljoin(ANIMETOSHO_FEED_URL, f"?t=search&q={title}")
    r = requests.get(query_url)
    j = r.json()

    # Loop over, make sure the link matches the URL and get a torrent link out
    parsed_url = None
    for i in j:

        if parsed_url is not None:
            continue

        link = i.get("link", None)
        if link == url:
            parsed_url = i.get("torrent_url", None)

    return parsed_url, title


def get_rutracker_torrent(
    url: str,
    torrent_hash: str,
) -> tuple[str, str]:
    """Get the RuTracker magnet link and torrent title from a URL

    Args:
        url (str): URL of the RuTracker topic
        torrent_hash (str): Torrent hash

    Returns:
        tuple: (magnet_url, torrent_title) - the magnet link and the
            human-readable torrent title scraped from the page
    """

    # Pull the torrent title from souping the URL
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "lxml")
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
