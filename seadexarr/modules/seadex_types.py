"""The ``seadex_dict`` domain vocabulary: the shapes the planner and notifier read.

The central ``seadex_dict`` is a four-level mapping built once per AniList entry
in :meth:`seadexarr.modules.seadex_arr.SeaDexArr.get_seadex_dict` (around
seadex_arr.py:368-376) and threaded through the decision engine
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

from dataclasses import dataclass, field
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

        raw_file = raw.get("episodeFile")
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


class Quality(TypedDict):
    """The nested ``quality`` object of a Sonarr ``QualityModel``.

    Schema ``Quality``: ``id``/``resolution`` are non-null ints, ``name`` is
    ``string | null``, ``source`` is the ``QualitySource`` enum (a string). Every
    key is ``NotRequired`` because the helpers build and read partial quality
    dicts (``resolve_quality_model`` copies only the keys a definition carries).
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
    read for its nested quality name and, when it wins the layered decision,
    re-emitted verbatim into the outgoing file payload; and
    ``resolve_quality_model`` builds one from a quality definition. Both keys are
    ``NotRequired`` because a built model may omit ``revision`` and a candidate
    model may carry only ``quality``.
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
        quality = raw.get("quality")
        return cls(
            path=raw.get("path"),
            quality=cast("QualityModel", quality) if isinstance(quality, dict) else None,
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
    (``trackedDownloadState``: ``downloading`` / ``importPending`` / ...),
    ``status`` (``trackedDownloadStatus``: ``ok`` / ``warning`` / ``error``) and
    whether ``statusMessages`` is populated (a message array on a pending item
    signals trouble, not progress). ``state``/``status`` are the ``string | null``
    rendering of their schema enums. Parsed at the client boundary via
    :meth:`from_api`, mirroring :class:`SonarrEpisode`.
    """

    download_id: str | None = None
    state: str | None = None
    status: str | None = None
    has_messages: bool = False

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``QueueResource`` dict (filters unknown keys).

        ``statusMessages`` is folded to a single ``has_messages`` bool here (a
        non-empty array means trouble), so the decision path never walks the
        message objects.
        """

        download_id = raw.get("downloadId")
        state = raw.get("trackedDownloadState")
        status = raw.get("trackedDownloadStatus")
        return cls(
            download_id=download_id if isinstance(download_id, str) else None,
            state=state if isinstance(state, str) else None,
            status=status if isinstance(status, str) else None,
            has_messages=bool(raw.get("statusMessages")),
        )


# --- Sonarr quality definitions (``/api/v3/qualitydefinition``) --------------


class QualityDefinition(TypedDict):
    """One Sonarr ``QualityDefinitionResource`` (schema), reduced to ``quality``.

    Read-and-re-emit: ``resolve_quality_model`` looks a name up against the nested
    ``quality.name`` and re-emits the matched :class:`Quality` verbatim into the
    outgoing ``QualityModel``, so a ``TypedDict`` types the JSON shape without a
    round-trip. Only the nested ``quality`` is consumed; ``quality`` is
    ``NotRequired`` because a malformed/partial definition may omit it (the
    resolver skips such entries).
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
class CommandResource:
    """A Sonarr ``CommandResource`` (schema), reduced to the fields read back.

    A command POST returns this with the queued command ``id``, and the status
    poll reads ``status`` (the ``CommandStatus`` enum: ``queued`` / ``started`` /
    ``completed`` / ...) to know when a rescan has settled. ``id`` is a non-null
    schema int (``0`` when absent so the caller drops it); ``status`` /
    ``result`` are the ``string | null`` rendering of their schema enums. Parsed
    at the client boundary via :meth:`from_api`, mirroring :class:`SonarrEpisode`.
    """

    id: int = 0
    status: str | None = None
    result: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Build from one raw ``CommandResource`` dict (filters unknown keys)."""

        status = raw.get("status")
        result = raw.get("result")
        return cls(
            id=raw.get("id", 0),
            status=status if isinstance(status, str) else None,
            result=result if isinstance(result, str) else None,
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
