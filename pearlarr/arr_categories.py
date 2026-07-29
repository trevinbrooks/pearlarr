"""The run's effective qBittorrent categories: config first, then the arr's own client.

A category left blank in config adopts the matching category of the arr's own
qBittorrent download client (`/api/v3/downloadclient`), so grabs and
post-import moves line up with what the arr does for downloads it starts
itself. Resolved once per arr run by the composition root (`RunDeps.build`);
every miss fails open to the blank.
"""

import logging
from dataclasses import dataclass

from .arr_http import ArrHttp
from .config import Arr, ArrSettings
from .log import LOG_NAME

# Debug breadcrumbs ride the stdlib channel (first-party child of the app
# logger), matching `arr_http`'s coalesced-repeat idiom.
_LOG = logging.getLogger(f"{LOG_NAME}.arr_categories")

# The arr-side settings field names, per arr: (add-time category, post-import
# category). CamelCased off the arrs' QBittorrentSettings properties.
_CATEGORY_FIELDS: dict[Arr, tuple[str, str]] = {
    Arr.SONARR: ("tvCategory", "tvImportedCategory"),
    Arr.RADARR: ("movieCategory", "movieImportedCategory"),
}


@dataclass(frozen=True)
class ArrCategories:
    """One arr run's resolved categories (None where nothing applies)."""

    grab: str | None
    """Applied to torrents added for this arr (`TorrentService`)."""

    post_import: str | None
    """Applied once a torrent's imports all complete (`ImportWaitManager`)."""


def resolve_arr_categories(arr: Arr, config: ArrSettings, http: ArrHttp | None) -> ArrCategories:
    """Resolve the effective categories, fetching the arr's download clients at most once.

    Per category: an explicit config value wins, a blank one adopts the arr's
    own qBittorrent download-client value. With `http` None (no qBittorrent
    client to apply a category, or missing connection keys), or on any fetch
    miss, blanks stay blank - the pre-fallback behavior.
    """

    grab = config.torrent_category or None
    post_import = config.post_import_category or None
    if http is None or (grab and post_import):
        return ArrCategories(grab=grab, post_import=post_import)
    client_grab, client_post = _client_categories(arr, http)
    return ArrCategories(grab=grab or client_grab, post_import=post_import or client_post)


def _client_categories(arr: Arr, http: ArrHttp) -> tuple[str | None, str | None]:
    """The (grab, post-import) categories of the arr's first enabled qBittorrent client.

    `(None, None)` when the fetch failed (the transport already warned), no
    enabled qBittorrent client is defined, or the fields are blank - each
    fail-open, logged once at DEBUG below.
    """

    clients = http.download_clients()
    if clients is None:
        return None, None
    client = next(
        (record for record in clients if record.enable and (record.implementation or "").casefold() == "qbittorrent"),
        None,
    )
    if client is None:
        _LOG.debug(f"{http.label} defines no enabled qBittorrent download client - blank categories stay blank")
        return None, None
    grab_field, post_field = _CATEGORY_FIELDS[arr]
    grab, post_import = client.field_value(grab_field), client.field_value(post_field)
    _LOG.debug(f"{http.label} download-client categories: {grab_field}={grab!r}, {post_field}={post_import!r}")
    return grab, post_import
