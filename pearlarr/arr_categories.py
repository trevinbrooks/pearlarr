"""The run's effective qBittorrent categories: config first, then the arr's own client.

A category OMITTED in config adopts the matching category of the arr's own
qBittorrent download client (`/api/v3/downloadclient`); an explicit blank
string keeps no category at all. Resolution is lazy - the client fetch happens
at first use (the run's first title checked, an import poll, a post-import
move), where the arr is provably up - and every miss fails open to blank.

The same read says which categories the arr watches: each of its enabled
qBittorrent clients lists only the torrents in its own category, and in that
category's subcategories while qBittorrent has subcategories turned on.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, final

import qbittorrentapi

from .arr_http import ArrHttp
from .arr_models import DownloadClientRecord
from .config import Arr, ArrSettings
from .log import LOG_NAME

# Debug breadcrumbs ride the stdlib channel (first-party child of the app
# logger), matching `arr_http`'s coalesced-repeat idiom.
_LOG = logging.getLogger(f"{LOG_NAME}.arr_categories")


class Watch(Enum):
    """Whether the arr's qBittorrent clients list a torrent. The arr tracks and imports only what they list."""

    WATCHED = auto()
    """A client lists it, or may: one with a blank category lists every torrent, and with no qBittorrent
    client found the arr may be reading qBittorrent some way this read cannot see."""

    UNWATCHED = auto()
    """No client lists it."""

    UNREAD = auto()
    """Not known: the arr's clients, the torrent or qBittorrent's subcategory setting could not be read."""


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


@dataclass(frozen=True, slots=True)
class CategorySources:
    """The two clients the categories are read from. A run has both or neither."""

    http: ArrHttp
    """The arr's transport, for its download clients."""
    qbit: qbittorrentapi.Client
    """Asked which category holds a torrent, and whether subcategories are on."""


@dataclass(frozen=True, slots=True)
class _ClientCategories:
    """One download-client read: the preferred client's categories, and every category the arr watches."""

    pair: _CategoryPair[str | None]
    """The (grab, post-import) categories of the arr's preferred qBittorrent client."""
    watched: frozenset[str] | None
    """The grab category of every enabled qBittorrent client.

    None when no category can be called unwatched: a client with a blank category lists every torrent, and with
    no qBittorrent client found the arr may be reading qBittorrent some way this read cannot see.
    """


def _explicit(value: str | None) -> str | None:
    """An explicit config category: blank/whitespace-only is the opt-out (no category, no fallback)."""

    return value if value and value.strip() else None


@final
class ArrCategoryResolver:
    """The two effective categories and which categories the arr watches, off one lazy client fetch a run.

    Lazy on purpose: at first use the arr is provably up (the library fetch before the run's first title, an
    import scan or a verified import just went through its API), where an eager boot-time fetch would turn an arr
    restart into a run-long miss. A successful fetch is memoized; a failed one is NOT - the next use retries, so a
    blip costs only the work racing it. Every miss fails open to blank (the transport warns, coalesced).
    """

    def __init__(self, arr: Arr, config: ArrSettings, sources: CategorySources | None) -> None:
        """`sources` None (no qBittorrent to apply a category, or missing connection keys) stays blank fetchless."""

        self._arr = arr
        # None = the key is absent from config, deferring to the arr's client.
        self._configured = _CategoryPair(config.torrent_category, config.post_import_category)
        self._sources = sources
        self._fetched: _ClientCategories | None = None
        self._nested: bool | None = None

    def grab(self) -> str | None:
        """The category for torrents added for this arr (`TorrentService`)."""

        value = self._configured.grab
        return self._client_pair().grab if value is None else _explicit(value)

    def post_import(self) -> str | None:
        """The category applied once a torrent's imports all complete (`ImportWaitManager`)."""

        value = self._configured.post_import
        return self._client_pair().post_import if value is None else _explicit(value)

    def download_categories(self, *, with_post_import: bool) -> tuple[str, ...] | None:
        """The grab category, and the post-import one when asked, for listing their torrents: no blank, each once.

        None while one of them is left to the arr's download clients and those can't be read.
        """

        configured = self._configured
        deferred = configured.grab is None or (with_post_import and configured.post_import is None)
        if deferred and self._clients() is None:
            return None
        names = (self.grab(), self.post_import() if with_post_import else None)
        return tuple(dict.fromkeys(name for name in names if name))

    def move_untracks(self, category: str) -> bool:
        """Whether moving a torrent to `category` takes it out of the categories the arr watches.

        The fetched clients are authoritative (a blank category means the arr watches everything).
        On a failed fetch the configured grab category approximates them, deliberately inverting
        `grab()`'s configured-first precedence. Unknown stays False.
        """

        clients = self._clients()
        if clients is not None:
            watched = clients.watched
        else:
            configured = _explicit(self._configured.grab)
            watched = None if configured is None else frozenset({configured})
        return self._watch(watched, category) is Watch.UNWATCHED

    def watch(self, category: str) -> Watch:
        """Whether the arr's clients list a torrent in `category`.

        Read from the arr's own clients, never the configured category. A wrong `UNWATCHED` gets a download
        imported here that the arr then imports again by its own parse.
        """

        clients = self._clients()
        return Watch.UNREAD if clients is None else self._watch(clients.watched, category)

    def torrent_watch(self, infohash: str) -> Watch:
        """Whether the arr's clients list a torrent, by the category qBittorrent holds it in now.

        `UNREAD` for a torrent qBittorrent doesn't hold.
        """

        try:
            category = self.torrent_category(infohash)
        except qbittorrentapi.APIError:
            return Watch.UNREAD
        return Watch.UNREAD if category is None else self.watch(category)

    def torrent_category(self, infohash: str) -> str | None:
        """The category qBittorrent holds a torrent in, `""` for an uncategorized one.

        None when qBittorrent has no such torrent, or its row carries no readable category. A failed read
        raises `qbittorrentapi.APIError`.
        """

        if self._sources is None:
            return None
        rows = self._sources.qbit.torrents_info(torrent_hashes=infohash)
        raw = getattr(rows[0], "category", None) if rows else None
        return raw if isinstance(raw, str) else None

    def _watch(self, watched: frozenset[str] | None, category: str) -> Watch:
        """Whether qBittorrent's filter for one of the `watched` categories returns a torrent in `category`.

        It returns the category's own torrents, and those of its subcategories (`tv/done` under `tv`) only
        while qBittorrent's subcategories are on. `watched` None stands for a filter that returns every
        torrent. Exact compare: categories are case-sensitive.
        """

        if watched is None or category in watched:
            return Watch.WATCHED
        if not any(category.startswith(f"{name}/") for name in watched):
            return Watch.UNWATCHED
        nested = self._subcategories_on()
        if nested is None:
            return Watch.UNREAD
        return Watch.WATCHED if nested else Watch.UNWATCHED

    def _subcategories_on(self) -> bool | None:
        """Whether qBittorrent's subcategories are on, read once a run. None when the setting can't be read.

        A qBittorrent that doesn't report the setting counts as on: a subcategory then reads as watched, so
        nothing in one is imported here on the strength of a guess.
        """

        if self._nested is None:
            if self._sources is None:
                return None
            try:
                self._nested = self._sources.qbit.app_preferences().get("use_subcategories") is not False
            except qbittorrentapi.APIError:
                # Not memoized: the next use retries.
                return None
        return self._nested

    def _client_pair(self) -> _CategoryPair[str | None]:
        clients = self._clients()
        return _CategoryPair(None, None) if clients is None else clients.pair

    def _clients(self) -> _ClientCategories | None:
        """The arr's download-client read, memoized on success. None with no sources or on a failed fetch."""

        if self._fetched is None:
            if self._sources is None:
                return None
            clients = self._sources.http.download_clients()
            if clients is None:
                # The transport already warned. Not memoized: the next use retries.
                return None
            self._fetched = _read_clients(self._arr, clients)
        return self._fetched


def _read_clients(arr: Arr, clients: list[DownloadClientRecord]) -> _ClientCategories:
    """The preferred qBittorrent client's (grab, post-import) categories, and every enabled one's grab category.

    Preferred = enabled, lowest `priority` number (1 is the arrs' highest and
    default), list order breaking ties. Which client the arr itself would pick
    is not decidable from outside (round-robin within a priority group,
    indexer pins, tag restrictions) - the config docs say so. `(None, None)`
    when no enabled qBittorrent client is defined or the fields are blank,
    logged once at DEBUG. The arr polls every enabled client, so all of them count as watching.
    """

    candidates = [
        record for record in clients if record.enable and (record.implementation or "").casefold() == "qbittorrent"
    ]
    if not candidates:
        _LOG.debug(f"{arr.capitalize()} defines no enabled qBittorrent download client - blank categories stay blank")
        return _ClientCategories(_CategoryPair(None, None), None)
    client = min(candidates, key=lambda record: record.priority)
    fields = _CATEGORY_FIELDS[arr]
    pair = _CategoryPair(client.field_value(fields.grab), client.field_value(fields.post_import))
    _LOG.debug(
        f"{arr.capitalize()} download-client categories: "
        f"{fields.grab}={pair.grab!r}, {fields.post_import}={pair.post_import!r}"
    )
    grabs = [record.field_value(fields.grab) for record in candidates]
    named = frozenset(grab for grab in grabs if grab is not None)
    # A client with a blank category lists every torrent.
    return _ClientCategories(pair, named if None not in grabs else None)
