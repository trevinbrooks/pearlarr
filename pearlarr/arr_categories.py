"""The run's effective qBittorrent categories: config first, then the arr's own client.

A category OMITTED in config adopts the matching category of the arr's own
qBittorrent download client (`/api/v3/downloadclient`); an explicit blank
string keeps no category at all. Resolution is lazy - the client fetch happens
at the first grab or post-import move, where the arr is provably up - and
every miss fails open to blank.
"""

import logging
from typing import NamedTuple, final

from .arr_http import ArrHttp
from .config import Arr, ArrSettings
from .log import LOG_NAME
from .seadex_types import DownloadClientRecord

# Debug breadcrumbs ride the stdlib channel (first-party child of the app
# logger), matching `arr_http`'s coalesced-repeat idiom.
_LOG = logging.getLogger(f"{LOG_NAME}.arr_categories")


class _CategoryPair[T](NamedTuple):
    """An (add-time grab, post-import) pair: the arr-side field names or their values."""

    grab: T
    post_import: T


# The arr-side settings field names, per arr. CamelCased off the arrs'
# QBittorrentSettings properties.
_CATEGORY_FIELDS: dict[Arr, _CategoryPair[str]] = {
    Arr.SONARR: _CategoryPair("tvCategory", "tvImportedCategory"),
    Arr.RADARR: _CategoryPair("movieCategory", "movieImportedCategory"),
}


def _explicit(value: str) -> str | None:
    """An explicit config category: blank/whitespace-only is the opt-out (no category, no fallback)."""

    return value if value.strip() else None


class PostImportMove(NamedTuple):
    """A retire's one category resolve: the target and whether moving there untracks the torrent."""

    category: str | None
    untracks: bool


NO_MOVE = PostImportMove(None, untracks=False)
"""The no-qBittorrent resolve: nothing to move, so no untrack skip either."""


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
        # None = the key is absent from config, deferring to the arr's client.
        self._configured = _CategoryPair(config.torrent_category, config.post_import_category)
        self._http = http
        self._fetched: _CategoryPair[str | None] | None = None

    def grab(self) -> str | None:
        """The category for torrents added for this arr (`TorrentService`)."""

        value = self._configured.grab
        return self._client_pair().grab if value is None else _explicit(value)

    def post_import(self) -> str | None:
        """The category applied once a torrent's imports all complete (`ImportWaitManager`)."""

        value = self._configured.post_import
        return self._client_pair().post_import if value is None else _explicit(value)

    def post_import_move(self) -> PostImportMove:
        """`post_import()` plus its untrack verdict, resolved ONCE per retire for both cleanup effects."""

        category = self.post_import()
        return PostImportMove(category, bool(category) and self.move_untracks(category))

    def move_untracks(self, category: str) -> bool:
        """Whether moving a torrent to `category` takes it out of the arr's watched grab category.

        The fetched client category is authoritative (blank means the arr watches everything).
        On a failed fetch the configured grab category approximates it, deliberately inverting
        `grab()`'s configured-first precedence. Unknown stays False.
        Exact compare: qBittorrent categories are case-sensitive.
        """

        pair = self._client_pair()
        watched = pair.grab if self._fetched is not None else _explicit(self._configured.grab or "")
        return bool(watched) and category != watched

    def _client_pair(self) -> _CategoryPair[str | None]:
        if self._fetched is None:
            if self._http is None:
                return _CategoryPair(None, None)
            clients = self._http.download_clients()
            if clients is None:
                # The transport already warned. Not memoized: the next use retries.
                return _CategoryPair(None, None)
            self._fetched = _pick_categories(self._arr, clients)
        return self._fetched


def _pick_categories(arr: Arr, clients: list[DownloadClientRecord]) -> _CategoryPair[str | None]:
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
        return _CategoryPair(None, None)
    client = min(candidates, key=lambda record: record.priority)
    fields = _CATEGORY_FIELDS[arr]
    pair = _CategoryPair(client.field_value(fields.grab), client.field_value(fields.post_import))
    _LOG.debug(
        f"{arr.capitalize()} download-client categories: "
        f"{fields.grab}={pair.grab!r}, {fields.post_import}={pair.post_import!r}"
    )
    return pair
