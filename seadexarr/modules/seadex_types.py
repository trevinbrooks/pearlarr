"""The ``seadex_dict`` domain vocabulary: the shapes the planner and notifier read.

The central ``seadex_dict`` is a four-level mapping built once per AniList entry
by :meth:`seadexarr.modules.seadex_filter.SeadexReleaseFilter.build` (reached via
the ``SeaDexArr.get_seadex_dict`` delegator) and threaded through the decision engine
(:mod:`seadexarr.modules.planner`) and the Discord notifier
(:mod:`seadexarr.modules.notify`). The two keyed levels stay plain ``dict``\\ s
(release groups keyed by name, urls keyed by url string), but the *value
records* at each level are modeled as :func:`dataclasses.dataclass`: a real
domain model with attribute access (``item.download`` rather than
``item["download"]``) and defaults that make a partially built record legal. Each
field carries a default because the records are filled in across construction
stages (``episodes``/``all_episodes`` are appended later by the episode parser,
``download`` is flipped per call), so a freshly built record need not pass every
field.

The defaults also encode one load-bearing distinction:
``SeadexReleaseGroupItem.all_episodes`` is ``None`` when no episode parsing ran
(Radarr movies) and an empty ``list`` when parsing ran but found nothing -
``get_same_files_groups`` keys off exactly that difference.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Any,
    NamedTuple,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    runtime_checkable,
)

from seadex import Tag, Tracker


@dataclass
class EpisodeRecord:
    """One parsed ``{season, episode, size}`` record for a SeaDex file.

    ``season``/``episode`` default to ``None`` (a record built without them - only
    seen in characterization tests - reduces to a never-matching ``(None, None)``
    key, which can never collide with a real Arr episode).
    """

    season: int | None = None
    episode: int | None = None
    size: int = 0


@dataclass
class SeadexUrlItem:
    """One SeaDex url record within a release group.

    ``tracker`` holds a SeaDex ``Tracker`` object (not a str); it defaults to
    ``None`` because the test builders don't supply one and no in-scope consumer
    reads it.
    """

    url: str = ""
    files: list[str] = field(default_factory=list[str])
    size: list[int] = field(default_factory=list[int])
    tracker: Tracker = Tracker.OTHER
    is_public: bool = True
    is_dual_audio: bool = False
    hash: str | None = None
    download: bool = False
    episodes: list[EpisodeRecord] = field(default_factory=list[EpisodeRecord])


@dataclass
class SeadexReleaseGroupItem:
    """One SeaDex release-group record, keyed by url under ``urls``.

    ``all_episodes`` is ``None`` until the episode parser has run:
    ``get_same_files_groups`` deliberately distinguishes ``None`` (no episode
    parsing, e.g. Radarr) from an empty list (parsing ran but found nothing).
    """

    urls: dict[str, SeadexUrlItem] = field(default_factory=dict[str, SeadexUrlItem])
    tags: frozenset[Tag] = field(default_factory=frozenset[Tag])
    all_episodes: list[EpisodeRecord] | None = None


SeadexDict = dict[str, SeadexReleaseGroupItem]
"""The central object: SeaDex release groups keyed by group name."""


class EmbedField(NamedTuple):
    """A Discord embed field, typed inside the notifier.

    The ``discordwebhook`` library wants plain ``{"name", "value"}`` dicts at the
    JSON boundary, so :meth:`to_dict` serializes the field back to that exact
    shape when :meth:`seadexarr.modules.notify.Notifier.build_fields` returns.
    """

    name: str
    value: str

    def to_dict(self) -> dict[str, str]:
        """The plain dict the Discord webhook payload expects."""

        return {"name": self.name, "value": self.value}


SONARR_MISSING_KEY: int = 999
"""Out-of-range fallback for a missing Sonarr ``seasonNumber``/``episodeNumber``.

Used when indexing Sonarr episodes by (season, episode); it never collides with
a real key, so an episode with a missing key simply fails to match.
"""


def season_episode_key(season: int | None, episode: int | None) -> tuple[int, int]:
    """The ``(season, episode)`` index key, collapsing a missing number to the sentinel.

    A missing ``season``/``episode`` collapses to :data:`SONARR_MISSING_KEY` (an
    out-of-range value that never collides with a real key), so our SeaDex
    ``(season, episode)`` and Sonarr's episode list key the same way. Shared by
    every ``(season, episode) -> ...`` index and lookup (the planner's episode
    index, ``build_episode_id_map``, the exact-parse lookup) so the sentinel
    convention lives in exactly one place.
    """

    return (
        season if season is not None else SONARR_MISSING_KEY,
        episode if episode is not None else SONARR_MISSING_KEY,
    )


def as_size_list(size: int | list[int | None] | None) -> list[int]:
    """Normalize a size value to a list of concrete sizes.

    ``None`` (or a missing size) becomes ``[]``; a bare int becomes ``[int]``; a
    list is copied with any ``None`` entries dropped (a ``None`` size carries no
    size to compare). The single home for the size-as-list coercion the planner
    used to inline.
    """

    if size is None:
        return []
    if isinstance(size, int):
        return [size]
    return [s for s in size if s is not None]


# --- shared progress sink ----------------------------------------------------


class ProgressSink(Protocol):
    """Sink for step progress - drives the boot cockpit's live bar.

    Structural, so the boot view's step handle satisfies it without the data /
    gateway modules importing the UI layer. ``fraction`` is 0-1 completion;
    ``detail`` is a short human note.
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...


# --- Arr items (Sonarr series / Radarr movies) ------------------------------


@runtime_checkable
class ArrItem(Protocol):
    """The attribute surface shared by a Sonarr series and a Radarr movie."""

    id: int
    title: str
    imdbId: str | None
    monitored: bool


@runtime_checkable
class SonarrItem(ArrItem, Protocol):
    """A Sonarr series item: an :class:`ArrItem` keyed on ``tvdbId``."""

    tvdbId: int


@runtime_checkable
class RadarrItem(ArrItem, Protocol):
    """A Radarr movie item: an :class:`ArrItem` keyed on ``tmdbId``."""

    tmdbId: int


# --- Sonarr episodes (``/api/v3/episode`` JSON) -----------------------------


@dataclass(frozen=True, slots=True)
class SonarrEpisodeFile:
    """The ``episodeFile`` sub-record of a Sonarr episode."""

    release_group: str | None = None
    size: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw Sonarr ``episodeFile`` dict."""

        return cls(release_group=raw.get("releaseGroup"), size=raw.get("size"))


@dataclass(frozen=True, slots=True)
class SonarrEpisode:
    """One Sonarr ``/api/v3/episode`` record, parsed at the client boundary."""

    id: int = 0
    season_number: int | None = None
    episode_number: int | None = None
    episode_file_id: int = 0
    monitored: bool = True
    episode_file: SonarrEpisodeFile | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw Sonarr episode dict (filters unknown keys)."""

        raw_file: dict[str, Any] | None = raw.get("episodeFile")
        return cls(
            id=raw.get("id", 0),
            season_number=raw.get("seasonNumber"),
            episode_number=raw.get("episodeNumber"),
            episode_file_id=raw.get("episodeFileId", 0),
            monitored=raw.get("monitored", True),
            episode_file=SonarrEpisodeFile.from_api(raw_file) if raw_file else None,
        )


type ArrReleaseDict = dict[str | None, list[int | None]]
"""Release group (``None`` when unknown) -> its existing-file sizes.

Built by the strategies (Sonarr accumulates a per-episode size list; Radarr
wraps its single movie size in a one-element list) and read in the planner via
:func:`as_size_list`, which drops the ``None`` placeholders.
"""


type TvdbMappings = dict[int, list[tuple[int, int | None]]]
"""AniBridge TVDB season -> inclusive ``(start, end)`` episode ranges."""


# --- AniList GraphQL errors (the ``errors`` array of a response body) --------


@dataclass(frozen=True, slots=True)
class AniListError:
    """One entry of an AniList GraphQL ``errors`` array, parsed at the boundary.

    AniList follows the GraphQL error shape and adds a numeric ``status`` (an
    HTTP-style code, e.g. ``429`` when soft-throttling). Only ``status`` and
    ``message`` drive the retry decision, so those are the fields modeled here.
    """

    message: str = ""
    status: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw GraphQL error dict (missing keys default)."""

        status = raw.get("status")
        return cls(
            message=str(raw.get("message") or ""),
            status=status if isinstance(status, int) else None,
        )


# --- AniList Media node (cached GraphQL ``Media`` record) --------------------


@dataclass(frozen=True, slots=True)
class AniListMediaNode:
    """One AniList ``Media`` node, parsed once at the cache read boundary."""

    id: int | None = None
    title_english: str | None = None
    title_romaji: str | None = None
    episodes: int | None = None
    cover_image: str | None = None
    format: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from a raw AniList ``Media`` dict (``{}`` on a miss).

        Reads the nested ``title``/``coverImage`` sub-objects null-safely,
        mirroring the former chained ``.get(...) or {}`` walks: an English title
        is preferred with a romaji fallback handled by the caller, and the
        ``large`` cover variant is the one downstream reads.
        """

        title: dict[str, Any] = raw.get("title") or {}
        cover: dict[str, Any] = raw.get("coverImage") or {}
        return cls(
            id=raw.get("id"),
            title_english=title.get("english"),
            title_romaji=title.get("romaji"),
            episodes=raw.get("episodes"),
            cover_image=cover.get("large"),
            format=raw.get("format"),
        )


# --- Sonarr manual-import (candidate read views + outgoing file payload) -----
#
# Derived from the Sonarr v3 OpenAPI ``ManualImportResource`` (and its nested
# ``QualityModel`` / ``Quality`` / ``Revision`` / ``Language`` /
# ``ImportRejectionResource``) in ``schemas/sonarr.schema``. Nullability mirrors
# the schema exactly (a schema ``string | null`` field -> ``str | None``).
#
# Two kinds of model live here, per the repo's hybrid rule:
#   * ``Quality`` / ``Revision`` / ``QualityModel`` / ``Language`` are TypedDicts:
#     a candidate's in-context ``QualityModel`` is read for a name AND re-emitted
#     verbatim into the outgoing payload, and a resolved ``Language`` is built as
#     a dict and POSTed, so a TypedDict types the JSON shape without forcing a
#     round-trip through a dataclass. The ``Quality`` / ``Revision`` /
#     ``QualityModel`` keys are ``NotRequired`` because the helpers build and read
#     *partial* objects (a candidate ``QualityModel`` carrying only ``quality``
#     and no ``revision``, a quality dict with just ``id``/``name``); ``Language``
#     always carries both ``id`` and ``name`` (nullable values from a ``.get()``).
#   * ``ManualImportCandidate`` / ``ImportRejection`` are frozen dataclass VIEWS
#     with a defensive ``from_api``: they are READ into the import decision, so
#     they follow the ``SonarrEpisode`` precedent (attribute access, ``.get()``
#     defaults).


class QualitySource(StrEnum):
    """Sonarr's ``QualitySource`` enum (schema ``QualitySource``).

    The structured ``source`` axis of a :class:`Quality`, modeled verbatim from
    the Sonarr OpenAPI schema (``schemas/sonarr.schema``) - the values are
    camelCase strings as Sonarr serializes them. Quality is matched on the
    ``(source, resolution)`` pair (never on the display name), so this enum is the
    authoritative source vocabulary the manual-import quality decision works in.
    ``BLURAY_RAW`` is a BD remux; ``TELEVISION_RAW`` is Raw-HD.
    """

    UNKNOWN = "unknown"
    TELEVISION = "television"
    TELEVISION_RAW = "televisionRaw"
    WEB = "web"
    WEBRIP = "webRip"
    DVD = "dvd"
    BLURAY = "bluray"
    BLURAY_RAW = "blurayRaw"

    @classmethod
    def parse(cls, value: str | None) -> "QualitySource | None":
        """A real source from a raw enum string, or None when undetermined.

        Case-insensitive. Returns None for a missing value, an unrecognized
        string, or ``"unknown"`` - i.e. None means "no authoritative source", so
        the caller's next precedence layer (our parse, then the configured
        default) gets a chance to fill the axis.

        Args:
            value (str | None): A raw ``QualitySource`` string from Sonarr JSON.

        Returns:
            QualitySource | None: The matched source, or None when undetermined.
        """

        if not value:
            return None
        folded = value.casefold()
        for member in cls:
            if member is cls.UNKNOWN:
                continue
            if member.value.casefold() == folded:
                return member
        return None


class Quality(TypedDict):
    """The nested ``quality`` object of a Sonarr ``QualityModel``.

    Schema ``Quality``: ``id``/``resolution`` are non-null ints, ``name`` is
    ``string | null``, ``source`` is the :class:`QualitySource` enum (a string).
    Every key is ``NotRequired`` because the helpers build and read partial
    quality dicts (the quality resolver copies only the keys a definition carries).
    """

    id: NotRequired[int]
    name: NotRequired[str | None]
    source: NotRequired[str]
    resolution: NotRequired[int]


class Revision(TypedDict):
    """The nested ``revision`` object of a Sonarr ``QualityModel`` (schema ``Revision``)."""

    version: NotRequired[int]
    real: NotRequired[int]
    isRepack: NotRequired[bool]


class QualityModel(TypedDict):
    """A Sonarr ``QualityModel`` (schema): ``{quality, revision}``.

    Used two ways on the manual-import path: a candidate's in-context model is
    read for its structured ``quality.source``/``quality.resolution`` axes and,
    when no definition matches the resolved quality, re-emitted verbatim into the
    outgoing file payload; and ``resolve_quality`` builds one from the matched
    quality definition. Both keys are ``NotRequired`` because a built model may
    omit ``revision`` and a candidate model may carry only ``quality``.
    """

    quality: NotRequired[Quality]
    revision: NotRequired[Revision]


class Language(TypedDict):
    """A Sonarr ``Language`` (schema): ``{id, name}``.

    Read+passthrough: ``resolve_language_objects`` builds these from the
    ``/api/v3/language`` list and they are POSTed verbatim in the file payload.
    ``id``/``name`` come from a defensive ``.get()``, so both are nullable.
    """

    id: int | None
    name: str | None


@dataclass(frozen=True, slots=True)
class ImportRejection:
    """One entry of a candidate's ``rejections`` array (schema ``ImportRejectionResource``).

    Only ``reason`` (``string | null``) is read - it carries the human text the
    sample / already-imported classifier matches against.
    """

    reason: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``ImportRejectionResource`` dict."""

        return cls(reason=raw.get("reason"))


@dataclass(frozen=True, slots=True)
class ManualImportCandidate:
    """A Sonarr ``ManualImportResource``, reduced to the fields planning reads.

    The decision path consults only ``path`` (the on-disk file to import,
    ``string | null`` in the schema), ``quality`` (the in-context ``QualityModel``
    re-emitted verbatim when it wins) and ``rejections`` (the per-file
    sample/already-imported flags). Parsed at the client boundary via
    :meth:`from_api`, mirroring :class:`SonarrEpisode`.
    """

    path: str | None = None
    quality: QualityModel | None = None
    rejections: tuple[ImportRejection, ...] = ()

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``ManualImportResource`` dict (filters unknown keys).

        ``rejections`` may be ``null`` (schema) and, on older Sonarr versions, a
        bare string per entry rather than an ``ImportRejectionResource`` object;
        both are folded to an :class:`ImportRejection` so the classifier reads one
        shape. ``quality`` is kept as the raw ``QualityModel`` mapping (re-emitted
        verbatim), so it is passed through unchanged.
        """

        rejections: list[ImportRejection] = []
        raw_rejections: list[Any] = raw.get("rejections") or []
        for rejection in raw_rejections:
            if isinstance(rejection, str):
                rejections.append(ImportRejection(reason=rejection))
            elif isinstance(rejection, dict):
                rejections.append(
                    ImportRejection.from_api(cast("dict[str, Any]", rejection)),
                )

        # The candidate's in-context QualityModel is re-emitted verbatim into the
        # outgoing payload, so it is passed through as the raw mapping; narrow it
        # to QualityModel at this parse boundary.
        quality: Mapping[str, Any] | None = raw.get("quality")
        return cls(
            path=raw.get("path"),
            quality=cast(QualityModel, quality) if quality else None,
            rejections=tuple(rejections),
        )


class ManualImportFile(TypedDict):
    """One outgoing ``ManualImport`` command file entry (the POST payload).

    Built by the Sonarr strategy from a planned ``import`` decision and POSTed as
    JSON, so a ``TypedDict`` types the constructed dict without a round-trip.
    ``quality`` is the only optional key (omitted, not sent as ``None``, when no
    quality resolves - Sonarr then falls back to Unknown).
    """

    path: str
    seriesId: int
    episodeIds: list[int]
    releaseGroup: str
    downloadId: str
    languages: list[Language]
    quality: NotRequired[QualityModel]


# --- Sonarr queue (``/api/v3/queue`` records) -------------------------------
#
# Derived from the Sonarr v3 OpenAPI ``QueueResource`` in ``schemas/sonarr.schema``.
# The endpoint pages its records under a wrapper object's ``records`` array.


@dataclass(frozen=True, slots=True)
class QueueRecord:
    """One Sonarr ``QueueResource`` record, reduced to the fields the wait reads.

    The wait/import decision consults only ``download_id`` (the infohash Sonarr
    stores uppercased, matched case-insensitively to pick a torrent's records -
    ``string | null`` in the schema), ``state``
    (``trackedDownloadState``: ``downloading`` / ``importPending`` / ...) and
    ``status`` (``trackedDownloadStatus``: ``ok`` / ``warning`` / ``error``).
    ``state``/``status`` are the ``string | null`` rendering of their schema
    enums. Parsed at the client boundary via :meth:`from_api`, mirroring
    :class:`SonarrEpisode`.
    """

    download_id: str | None = None
    state: str | None = None
    status: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``QueueResource`` dict (filters unknown keys)."""

        download_id = raw.get("downloadId")
        state = raw.get("trackedDownloadState")
        status = raw.get("trackedDownloadStatus")
        return cls(
            download_id=download_id if isinstance(download_id, str) else None,
            state=state if isinstance(state, str) else None,
            status=status if isinstance(status, str) else None,
        )


# --- Sonarr quality definitions (``/api/v3/qualitydefinition``) --------------


class QualityDefinition(TypedDict):
    """One Sonarr ``QualityDefinitionResource`` (schema), reduced to ``quality``.

    Read-and-re-emit: ``resolve_quality`` matches a definition by its nested
    ``quality.source``/``quality.resolution`` pair and re-emits the matched
    :class:`Quality` verbatim into the outgoing ``QualityModel``, so a
    ``TypedDict`` types the JSON shape without a round-trip. Only the nested
    ``quality`` is consumed; ``quality`` is ``NotRequired`` because a
    malformed/partial definition may omit it (the resolver skips such entries).
    """

    quality: NotRequired[Quality]


# --- Sonarr commands (``/api/v3/command``) -----------------------------------


class CommandBody(TypedDict):
    """One outgoing ``/api/v3/command`` POST body (a Sonarr command request).

    Constructed by the strategy and POSTed as JSON, so a ``TypedDict`` types the
    body without a round-trip. ``name`` is the command name (always sent);
    ``importMode`` / ``files`` are the extra keys the ``ManualImport`` command
    carries, so both are ``NotRequired`` (``RefreshMonitoredDownloads`` sends only
    ``name``).
    """

    name: str
    importMode: NotRequired[str]
    files: NotRequired[list[ManualImportFile]]


@dataclass(frozen=True, slots=True)
class CommandFile:
    """One file of a ``ManualImport`` command's ``body.files[]`` (read back).

    Surfaced from the ``/api/v3/command`` list so the in-flight guard can tell
    whether an accepted-but-still-running ManualImport already covers a download:
    ``download_id`` is the primary match key (the infohash a queue-driven import
    carries, ``string | null`` in the schema - absent for a folder/season-pack
    import), with ``path`` and ``episode_ids`` as the fallback signals. Parsed at
    the client boundary via :meth:`from_api`.
    """

    path: str | None = None
    download_id: str | None = None
    series_id: int = 0
    episode_ids: tuple[int, ...] = ()

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw command ``body.files[]`` entry (filters unknown keys)."""

        path = raw.get("path")
        download_id = raw.get("downloadId")
        raw_ids: list[Any] = raw.get("episodeIds") or []
        episode_ids = tuple(i for i in raw_ids if isinstance(i, int))
        return cls(
            path=path if isinstance(path, str) else None,
            download_id=download_id if isinstance(download_id, str) else None,
            series_id=raw.get("seriesId", 0),
            episode_ids=episode_ids,
        )


@dataclass(frozen=True, slots=True)
class CommandResource:
    """A Sonarr ``CommandResource`` (schema), reduced to the fields read back.

    A command POST returns this with the queued command ``id``, and the status
    poll reads ``status`` (the ``CommandStatus`` enum: ``queued`` / ``started`` /
    ``completed`` / ...) to know when a rescan has settled. ``id`` is a non-null
    schema int (``0`` when absent so the caller drops it); ``status`` /
    ``result`` are the ``string | null`` rendering of their schema enums.

    The ``/api/v3/command`` LIST poll reads the extra fields for the in-flight
    ManualImport guard: ``name`` (the command name, e.g. ``ManualImport``),
    ``message`` (the progress text, e.g. ``"Processing file 4 of 8"`` /
    ``"Manually imported 10 files"`` - kept for a later wait-view enrichment) and
    ``files`` (the per-file rows from ``body.files``, each a :class:`CommandFile`
    carrying the download id / path / episode ids that say which download a
    still-running import covers). All default to empty so the POST/status callers
    that only read ``id``/``status``/``result`` are unaffected. Parsed at the
    client boundary via :meth:`from_api`, mirroring :class:`SonarrEpisode`.
    """

    id: int = 0
    status: str | None = None
    result: str | None = None
    name: str | None = None
    message: str | None = None
    files: tuple[CommandFile, ...] = ()

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``CommandResource`` dict (filters unknown keys).

        ``files`` lives under the nested ``body`` object (the original command
        request Sonarr echoes back), so it is read from ``body.files`` when present;
        the POST/status responses omit it, leaving ``files`` empty.
        """

        status = raw.get("status")
        result = raw.get("result")
        name = raw.get("name")
        message = raw.get("message")
        body: dict[str, Any] = raw.get("body") or {}
        raw_files: list[Any] = body.get("files") or []
        files = tuple(CommandFile.from_api(cast("dict[str, Any]", f)) for f in raw_files if isinstance(f, dict))
        return cls(
            id=raw.get("id", 0),
            status=status if isinstance(status, str) else None,
            result=result if isinstance(result, str) else None,
            name=name if isinstance(name, str) else None,
            message=message if isinstance(message, str) else None,
            files=files,
        )


# --- Radarr movie files (``/api/v3/moviefile`` records) ----------------------
#
# Derived from the Radarr v3 OpenAPI ``MovieFileResource`` in
# ``schemas/radarr.schema``. Nullability mirrors the schema exactly (a schema
# ``string | null`` field -> ``str | None``).


@dataclass(frozen=True, slots=True)
class MovieFile:
    """A Radarr ``MovieFileResource``, reduced to the fields the syncer reads.

    ``get_radarr_release_dict`` reads each movie file into the shared
    :data:`ArrReleaseDict` decision (release group -> existing-file sizes), so a
    movie file is READ into a decision, not re-emitted: it follows the
    :class:`SonarrEpisode` precedent (a frozen dataclass VIEW with a defensive
    :meth:`from_api`). Only ``release_group`` (``string | null`` in the schema)
    and ``size`` (a non-null ``int64``) are consumed. Parsed at the client
    boundary.
    """

    release_group: str | None = None
    size: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``MovieFileResource`` dict (filters unknown keys)."""

        return cls(
            release_group=raw.get("releaseGroup"),
            size=raw.get("size"),
        )


# --- Sonarr parse (``/api/v3/parse`` ``episodes`` array) ---------------------


class ParsedEpisode(TypedDict):
    """One entry of a Sonarr ``ParseResource`` ``episodes`` array (schema ``EpisodeResource``).

    The ``/api/v3/parse`` response nests an ``episodes`` array; only the
    season/episode numbers are read out of each entry (the file size comes from
    the SeaDex file list, not Sonarr). Both are ``NotRequired`` because a parse
    that couldn't resolve an entry may omit them (the reader drops such entries).
    """

    seasonNumber: NotRequired[int]
    episodeNumber: NotRequired[int]


class ParsedEpisodeInfo(TypedDict):
    """The ``parsedEpisodeInfo`` object of a Sonarr ``/api/v3/parse`` response.

    Unlike the response's ``episodes`` array (Sonarr's *series-matched* episodes,
    which is empty whenever the release title can't be matched to a series in the
    library), this object carries the numbers Sonarr parsed straight from the
    release NAME, independent of any series match: the season + episode numbers
    for an ``SxxExx`` name, and the absolute episode numbers for an
    absolute-numbered anime release. Every field is ``NotRequired`` because a name
    Sonarr couldn't parse may omit them.
    """

    seasonNumber: NotRequired[int]
    episodeNumbers: NotRequired[list[int]]
    absoluteEpisodeNumbers: NotRequired[list[int]]
    special: NotRequired[bool]


@dataclass(frozen=True, slots=True)
class ParsedFileInfo:
    """Sonarr's series-AGNOSTIC parse of one filename, narrowed to what assignment reads.

    Built from a ``/api/v3/parse`` ``parsedEpisodeInfo`` at the Sonarr client
    boundary. ``episode_numbers`` (paired with ``season_number``) drives the exact
    ``(season, episode)`` assignment; ``absolute_episode_numbers`` drives the
    absolute-index fallback. Both are read straight from the release name, so they
    are populated even when Sonarr can't match the title to a series - which is
    exactly the case (specials, alias titles) the old series-matched parse failed.

    ``season_number`` is whatever Sonarr reported and is meaningful only when
    ``episode_numbers`` is non-empty (an absolute-numbered name reports season 0).
    """

    season_number: int | None = None
    episode_numbers: tuple[int, ...] = ()
    absolute_episode_numbers: tuple[int, ...] = ()
    special: bool = False

    @classmethod
    def from_parse_resource(cls, body: dict[str, Any]) -> Self:
        """Build from a raw ``/api/v3/parse`` response body (``parsedEpisodeInfo``)."""

        info = cast("ParsedEpisodeInfo", body.get("parsedEpisodeInfo") or {})
        return cls(
            season_number=info.get("seasonNumber"),
            episode_numbers=tuple(info.get("episodeNumbers") or ()),
            absolute_episode_numbers=tuple(info.get("absoluteEpisodeNumbers") or ()),
            special=bool(info.get("special", False)),
        )
