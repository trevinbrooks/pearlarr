"""The run's effective qBittorrent categories: config first, then the arr's own client.

A category OMITTED in config adopts the matching category of the arr's own
qBittorrent download client (`/api/v3/downloadclient`); an explicit blank
string keeps no category at all. Resolution is lazy - the client fetch happens
at the first grab or post-import move, where the arr is provably up - and
every miss fails open to blank.
"""

import logging
from typing import final

from .arr_http import ArrHttp
from .config import Arr, ArrSettings
from .log import LOG_NAME
from .seadex_types import DownloadClientRecord

# Debug breadcrumbs ride the stdlib channel (first-party child of the app
# logger), matching `arr_http`'s coalesced-repeat idiom.
_LOG = logging.getLogger(f"{LOG_NAME}.arr_categories")

# The arr-side settings field names, per arr: (add-time category, post-import
# category). CamelCased off the arrs' QBittorrentSettings properties.
_CATEGORY_FIELDS: dict[Arr, tuple[str, str]] = {
    Arr.SONARR: ("tvCategory", "tvImportedCategory"),
    Arr.RADARR: ("movieCategory", "movieImportedCategory"),
}


def _configured(value: str | None) -> tuple[bool, str | None]:
    """One config category as `(settled, category)`.

    An explicit value pins it, a blank/whitespace-only string is the opt-out
    (no category, no fallback), and an absent key (None) defers to the arr's
    own download client.
    """

    if value is None:
        return False, None
    return True, (value if value.strip() else None)


@final
class ArrCategoryResolver:
    """Resolves the two effective categories lazily, one client fetch per run at most.

    Lazy on purpose: at first use the arr is provably up (a grab or a verified
    import just went through its API), where an eager boot-time fetch would
    turn an arr restart into a run-long miss. A successful fetch is memoized;
    a failed one is NOT - the next use retries, so a blip costs only the work
    racing it. Every miss fails open to blank (the transport warns, coalesced).
    """

    def __init__(self, arr: Arr, config: ArrSettings, http: ArrHttp | None) -> None:
        """`http` None (no qBittorrent to apply a category, or missing connection keys) stays blank fetchless."""

        self._arr = arr
        self._grab = _configured(config.torrent_category)
        self._post_import = _configured(config.post_import_category)
        self._http = http
        self._fetched: tuple[str | None, str | None] | None = None

    def grab(self) -> str | None:
        """The category for torrents added for this arr (`TorrentService`)."""

        settled, category = self._grab
        return category if settled else self._client_pair()[0]

    def post_import(self) -> str | None:
        """The category applied once a torrent's imports all complete (`ImportWaitManager`)."""

        settled, category = self._post_import
        return category if settled else self._client_pair()[1]

    def _client_pair(self) -> tuple[str | None, str | None]:
        if self._fetched is None:
            if self._http is None:
                return None, None
            clients = self._http.download_clients()
            if clients is None:
                # The transport already warned. Not memoized: the next use retries.
                return None, None
            self._fetched = _pick_categories(self._arr, clients)
        return self._fetched


def _pick_categories(arr: Arr, clients: list[DownloadClientRecord]) -> tuple[str | None, str | None]:
    """The (grab, post-import) categories of the arr's preferred qBittorrent client.

    Preferred = enabled, lowest `priority` number (1 is the arrs' highest and
    default), list order breaking ties. Which client the arr itself would pick
    is not decidable from outside (round-robin within a priority group,
    indexer pins, tag restrictions) - the config docs say so. `(None, None)`
    when no enabled qBittorrent client is defined or the fields are blank,
    logged once at DEBUG.
    """

    candidates = [
        record for record in clients if record.enable and (record.implementation or "").casefold() == "qbittorrent"
    ]
    if not candidates:
        _LOG.debug(f"{arr.capitalize()} defines no enabled qBittorrent download client - blank categories stay blank")
        return None, None
    client = min(candidates, key=lambda record: record.priority)
    grab_field, post_field = _CATEGORY_FIELDS[arr]
    grab, post_import = client.field_value(grab_field), client.field_value(post_field)
    _LOG.debug(f"{arr.capitalize()} download-client categories: {grab_field}={grab!r}, {post_field}={post_import!r}")
    return grab, post_import
