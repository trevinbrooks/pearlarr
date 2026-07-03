import time

import requests

# (connect, read) bound so a hung Discord webhook surfaces as a transient miss
# instead of blocking the run; matches the Arr clients' timeout policy.
DISCORD_TIMEOUT_S = (5, 30)


def discord_push(
    url: str,
    arr_title: str,
    al_title: str,
    seadex_url: str,
    fields: list[dict[str, str]],
    thumb_url: str | None,
) -> bool:
    """Post a message to Discord

    Raises ``requests.RequestException`` (incl. HTTP error statuses) so the
    caller's containment decides; a webhook failure must never abort a grab.

    Args:
        url (str): URL to post to
        arr_title (str): Title as in Arr instance
        al_title (str): Title as in AniList
        seadex_url (str): URL to SeaDex page
        fields (list): List of dicts containing links
            for the fields
        thumb_url (str | None): URL for thumbnail, if any
    """

    payload = {
        "embeds": [
            {
                "author": {
                    "name": arr_title,
                    "url": "https://github.com/bbtufty/seadexarr",
                },
                "title": al_title,
                "description": seadex_url,
                "fields": fields,
                "thumbnail": {"url": thumb_url},
            },
        ],
    }
    response = requests.post(url, json=payload, timeout=DISCORD_TIMEOUT_S)
    response.raise_for_status()

    # Sleep for a bit to avoid rate limiting
    time.sleep(1)

    return True
