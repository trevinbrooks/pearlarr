"""Domain vocabulary + the typed API boundary: the shapes a run reads and writes.

Two halves live here. The first is the `seadex_dict` domain vocabulary: the
central `seadex_dict` is a four-level mapping built once per AniList entry by
`SeadexReleaseFilter.build` (reached via the `RunServices.get_seadex_dict`
delegator) and threaded through the decision engine (`planner`) and the
Discord notifier (`notify`). The two keyed levels stay plain `dict`s
(release groups keyed by name, urls keyed by url string), but the *value
records* at each level are modeled as `dataclasses.dataclass`: a real
domain model with attribute access (`item.download` rather than
`item["download"]`) and defaults that make a partially built record legal. Each
field carries a default because the records are filled in across construction
stages (`episodes`/`all_episodes` are appended later by the episode parser,
`download` is flipped per call), so a freshly built record need not pass every
field.

The defaults also encode one load-bearing distinction:
`SeadexReleaseGroupItem.all_episodes` is `None` when no episode parsing ran
(Radarr movies) and an empty `list` when parsing ran but found nothing -
`get_same_files_groups` keys off exactly that difference.

The second half (from "pydantic boundary plumbing" down) is the typed API
boundary: pydantic models that arr/AniList JSON is validated into at the
client edge, replacing the former hand-plucked `from_api` dicts and
TypedDict casts. The regime:

* READ models subclass `_ApiModel` (frozen, unknown keys ignored) and
  enter via `validate_each` - the fail-open list read, one scrubbed
  warning per skipped record - or via `model_validate` inside the owning
  client's single-object fail-open try/except (degrading to the site's
  existing default, e.g. the all-None `AniListMediaNode` miss node).
* The library fetches (`SonarrSeries` / `RadarrMovie`) read
  STRICT: `validate_each(..., strict=True)` raises
  `BoundaryContractError` when a non-empty payload validates to
  nothing, so a broken endpoint never reads as an empty library.
* Records whose loss would change a decision fold junk PER FIELD instead of
  skipping (`QueueRecord`, `HistoryRecord`,
  `CommandResource`/`CommandFile`, via the reusable lenient
  validators): a dropped queue record could route a wait into a
  double-importing step-in; a dropped history record is a missed dirty-mark.
* WRITE / re-emit shapes subclass `_WireModel` (`extra="allow"`) so
  unknown keys survive the read->resolve->re-emit round-trip; the standard
  write dump is `model_dump(exclude_unset=True)`, never `exclude_none`.

The port also fixed two latent crashes: a junk `rejections` entry can no
longer reach the import classifier's `.casefold()`
(`ImportRejection` folds/skips them), and a JSON-array AniList body is
no longer laundered into a dict (it folds to the `{}` no-data arm in
`anilist_client`).

Kept deliberately un-ported: `SonarrParseRecord` (`sonarr_parse` - a
persisted parse-cache row, not a wire read), `coerce_int` (the lenient
int fold behind `_int_or_zero`, plus non-boundary coercions in
`anibridge` / `manual_import`), the `Json` alias (typing
for constructed payloads and the `json_narrow` guards), and the
structural protocols (`ArrItem` and friends).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from seadex import Tag, Tracker


@dataclass
class EpisodeRecord:
    """One parsed `{season, episode, size}` record for a SeaDex file.

    `season`/`episode` default to `None` (a record built without them - only
    seen in characterization tests - reduces to a never-matching `(None, None)`
    key, which can never collide with a real Arr episode).
    """

    season: int | None = None
    episode: int | None = None
    size: int = 0


@dataclass
class SeadexUrlItem:
    """One SeaDex url record within a release group.

    `tracker` holds a SeaDex `Tracker` object (not a str); the notifier
    renders it as the link text of a grab embed. It defaults to `OTHER`
    because the test builders don't supply one.
    """

    url: str = ""
    files: list[str] = field(default_factory=list[str])
    size: list[int] = field(default_factory=list[int])
    tracker: Tracker = Tracker.OTHER
    is_public: bool = True
    is_dual_audio: bool = False
    infohash: str | None = None
    download: bool = False
    # True for a public alternative added because the preferred release is
    # private-only (seadex.private_releases: fallback); the planner reads it.
    is_fallback: bool = False
    # True when the url was flagged because the Arr holds this release at a
    # different size (an upgrade), not because it lacks it.
    size_mismatch: bool = False
    episodes: list[EpisodeRecord] = field(default_factory=list[EpisodeRecord])


@dataclass
class SeadexReleaseGroupItem:
    """One SeaDex release-group record, keyed by url under `urls`.

    `all_episodes` is `None` until the episode parser has run:
    `get_same_files_groups` deliberately distinguishes `None` (no episode
    parsing, e.g. Radarr) from an empty list (parsing ran but found nothing).
    """

    urls: dict[str, SeadexUrlItem] = field(default_factory=dict[str, SeadexUrlItem])
    tags: frozenset[Tag] = field(default_factory=frozenset[Tag])
    all_episodes: list[EpisodeRecord] | None = None


SeadexDict = dict[str, SeadexReleaseGroupItem]
"""The central object: SeaDex release groups keyed by group name."""


SONARR_MISSING_KEY: int = 999
"""Out-of-range fallback for a missing Sonarr `seasonNumber`/`episodeNumber`.

Used when indexing Sonarr episodes by (season, episode); it never collides with
a real key, so an episode with a missing key simply fails to match.
"""


def season_episode_key(season: int | None, episode: int | None) -> tuple[int, int]:
    """The `(season, episode)` index key, collapsing a missing number to the sentinel.

    A missing `season`/`episode` collapses to `SONARR_MISSING_KEY` (an
    out-of-range value that never collides with a real key), so our SeaDex
    `(season, episode)` and Sonarr's episode list key the same way. Shared by
    every `(season, episode) -> ...` index and lookup (the planner's episode
    index, `build_episode_id_map`, the exact-parse lookup) so the sentinel
    convention lives in exactly one place.
    """

    return (
        season if season is not None else SONARR_MISSING_KEY,
        episode if episode is not None else SONARR_MISSING_KEY,
    )


def as_size_list(size: int | list[int | None] | None) -> list[int]:
    """Normalize a size value to a list of concrete sizes.

    `None` (or a missing size) becomes `[]`; a bare int becomes `[int]`; a
    list is copied with any `None` entries dropped (a `None` size carries no
    size to compare). The single home for the size-as-list coercion the planner
    used to inline.
    """

    if size is None:
        return []
    if isinstance(size, int):
        return [size]
    return [s for s in size if s is not None]


# --- shared plumbing ----------------------------------------------------------

# (connect, read) timeout shared by the arr httpx client factory and the
# qBittorrent adapter, so a hung service surfaces as a transient miss instead
# of blocking the run.
ARR_REQUEST_TIMEOUT_S = (5, 30)

# The recursive JSON value shape. Constructed JSON payloads are typed against
# this at the wire boundary (`ArrHttp.post_json`, the redact/narrow walks).
type Json = None | bool | int | float | str | Sequence["Json"] | Mapping[str, "Json"]


def coerce_int(value: object) -> int | None:
    """Best-effort int, or None for a non-numeric / NaN value.

    Ints pass through, floats convert unless NaN, strings via `int()`;
    anything else (including None) is None.
    """

    if isinstance(value, bool):
        return int(value)  # normalize True/False to 1/0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


# --- pydantic boundary plumbing ----------------------------------------------
#
# Every arr/AniList READ model below subclasses `_ApiModel` and is validated
# at the client boundary - list reads via `validate_each` (skip + scrubbed
# warn per bad record; `strict=True` for the load-bearing library fetches),
# single-object reads via `model_validate` in the owning client's fail-open
# try/except. Warnings NEVER embed payload values (only field locs + error
# types), so a malformed record can't leak titles/paths/keys into logs.


class _ApiModel(BaseModel):
    """Frozen boundary read model: unknown keys ignored, field-name kwargs allowed.

    `validate_by_name` is required: aliased fields are also constructed by
    field name across the tests/fakes, which would otherwise silently no-op to
    defaults. Bool policy: pydantic's lax coercion already preserves True -> 1
    on int fields, so no model rejects bools; at this boundary
    `coerce_int` survives only inside the `_int_or_zero` lenient fold.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", validate_by_name=True)


class _WireModel(BaseModel):
    """Frozen wire re-emit shape: unknown keys VALIDATE and RE-EMIT (extra="allow").

    For the read->resolve->re-emit round-trips (a candidate's quality model, a
    definition's nested quality) and our constructed write bodies. The standard
    write dump is `model_dump(exclude_unset=True)` - NEVER `exclude_none`
    (an explicitly-set None, e.g. a language's null id, must reach the wire) -
    so construction discipline applies: set every key the wire body needs.
    """

    model_config = ConfigDict(frozen=True, extra="allow")


class BoundaryContractError(RuntimeError):
    """A strict library read got a non-empty payload with zero valid records.

    Raised only by `validate_each(..., strict=True)` (the fail-CLOSED library
    fetches): an all-invalid library payload must abort the leg rather than
    read as an empty library. The CLI renders it via the same one-line
    containment arm as the typed `arr_http` connection errors.
    """


def validation_summary(e: ValidationError) -> str:
    """A log-safe summary of a validation failure: field locs + error types only.

    Deliberately built from `errors(include_input=False)` - never `str(e)`,
    which embeds the raw input values (payload data must not reach the logs).
    """

    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc']) or '<record>'}: {err['type']}"
        for err in e.errors(include_url=False, include_input=False)
    )


def validate_each[ModelT: _ApiModel](
    model: type[ModelT],
    raw: list[object],
    *,
    strict: bool = False,
) -> list[ModelT]:
    """Validate each raw record into `model`, skipping the ones that fail.

    The fail-open list read: every skipped record warns once, scrubbed
    (index + field locs + error types; never the payload). With `strict=True`
    a non-empty `raw` that validates to NOTHING raises
    `BoundaryContractError` instead of degrading to an empty list - the
    posture for the load-bearing library fetches, where "all records malformed"
    means the endpoint contract is broken, not that the library is empty.

    Args:
        model: The `_ApiModel` subclass to validate into.
        raw: The unvalidated JSON array elements.
        strict: Raise on a non-empty-but-zero-valid payload.
    """

    validated: list[ModelT] = []
    for index, record in enumerate(raw):
        try:
            validated.append(model.model_validate(record))
        except ValidationError as e:
            # Deferred: a top-level .output import would cycle back here via
            # output.events -> manual_import -> seadex_types. Skip-arm only,
            # so the all-valid hot path never touches the import machinery.
            from .output import hub_warn

            hub_warn(f"Skipping malformed {model.__name__} record [{index}] ({validation_summary(e)})")
    if strict and raw and not validated:
        msg = f"none of the {len(raw)} {model.__name__} records validated; refusing to treat it as empty"
        raise BoundaryContractError(msg)
    return validated


def _str_or_none(value: object) -> str | None:
    """Per-field lenient fold: keep a str, fold any other shape to None."""

    return value if isinstance(value, str) else None


def _str_or_blank(value: object) -> str:
    """Per-field lenient fold: keep a str, fold any other shape to ""."""

    return value if isinstance(value, str) else ""


def _int_or_zero(value: object) -> int:
    """Per-field lenient fold: best-effort int, folding junk/None to 0."""

    return coerce_int(value) or 0


def _stringified(value: object) -> str:
    """Per-field lenient fold: `str(value or "")` (a falsy value reads as "")."""

    return str(value or "")


def _none_if_falsy(value: object) -> object:
    """Fold a falsy value (`{}`/None) to None before nested validation."""

    return value or None


# Reusable lenient field shapes (the per-field folding regime).
type _LenientStr = Annotated[str | None, BeforeValidator(_str_or_none)]
type _BlankStr = Annotated[str, BeforeValidator(_str_or_blank)]
type _ZeroInt = Annotated[int, BeforeValidator(_int_or_zero)]


# --- shared progress sink ----------------------------------------------------


class ProgressSink(Protocol):
    """Sink for step progress - drives the boot cockpit's live bar.

    Structural, so the boot flow's step scope satisfies it without the data /
    gateway modules importing the output layer. `fraction` is 0-1 completion;
    `detail` is a short human note.
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...


# --- Arr items (Sonarr series / Radarr movies) ------------------------------


@runtime_checkable
class ArrItem(Protocol):
    """The attribute surface shared by a Sonarr series and a Radarr movie.

    Read-only properties (nothing writes to an item), so the frozen dataclass
    views (`SonarrSeries` / `RadarrMovie`) and mutable test
    stand-ins both satisfy it structurally.
    """

    @property
    def id(self) -> int: ...

    @property
    def title(self) -> str: ...

    @property
    def imdbId(self) -> str | None: ...

    @property
    def monitored(self) -> bool: ...


@runtime_checkable
class SonarrItem(ArrItem, Protocol):
    """A Sonarr series item: an `ArrItem` keyed on `tvdbId`."""

    @property
    def tvdbId(self) -> int: ...


@runtime_checkable
class RadarrItem(ArrItem, Protocol):
    """A Radarr movie item: an `ArrItem` keyed on `tmdbId`."""

    @property
    def tmdbId(self) -> int: ...


class SonarrSeries(_ApiModel):
    """One Sonarr `/api/v3/series` record, narrowed to the `SonarrItem` surface.

    The concrete item `SonarrClient.all_series` returns. camelCase field
    names on purpose: they satisfy the protocol directly, and the
    `IdField.item_attr` strings (`"tvdbId"`/`"imdbId"`) keep working.
    The STRICT library read (`validate_each(..., strict=True)`): a record
    with junk (or an explicit null) in a typed field is skipped with a warning,
    and an all-invalid non-empty payload raises `BoundaryContractError`
    rather than reading as an empty library.
    """

    id: int = 0
    title: str = ""
    monitored: bool = True
    tvdbId: int = 0
    imdbId: str | None = None


class RadarrMovie(_ApiModel):
    """One Radarr `/api/v3/movie` record, narrowed to the `RadarrItem` surface.

    The concrete item `RadarrClient.all_movies` returns, mirroring
    `SonarrSeries` (camelCase fields and the same strict library-read
    posture, for the same reasons).
    """

    id: int = 0
    title: str = ""
    monitored: bool = True
    tmdbId: int = 0
    imdbId: str | None = None


# --- Sonarr episodes (`/api/v3/episode` JSON) -----------------------------


class SonarrEpisodeFile(_ApiModel):
    """The `episodeFile` sub-record of a Sonarr episode."""

    release_group: str | None = Field(default=None, validation_alias="releaseGroup")
    size: int | None = None


class SonarrEpisode(_ApiModel):
    """One Sonarr `/api/v3/episode` record, validated at the client boundary.

    Fail-open list read: a record with junk in a typed field (its own or the
    nested `episodeFile`'s) is skipped with a warning by `validate_each`
    rather than flowing as a type lie.
    """

    id: int = 0
    season_number: int | None = Field(default=None, validation_alias="seasonNumber")
    episode_number: int | None = Field(default=None, validation_alias="episodeNumber")
    episode_file_id: int = Field(default=0, validation_alias="episodeFileId")
    monitored: bool = True
    # An empty/null episodeFile folds to None (the historical `if raw_file` gate).
    episode_file: Annotated[SonarrEpisodeFile | None, BeforeValidator(_none_if_falsy)] = Field(
        default=None,
        validation_alias="episodeFile",
    )

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(raw)


type ArrReleaseDict = dict[str | None, list[int | None]]
"""Release group (`None` when unknown) -> its existing-file sizes.

Built by the strategies (Sonarr accumulates a per-episode size list; Radarr
wraps its single movie size in a one-element list) and read in the planner via
`as_size_list`, which drops the `None` placeholders.
"""


type TvdbMappings = dict[int, list[tuple[int, int | None]]]
"""AniBridge TVDB season -> inclusive `(start, end)` episode ranges."""


# --- AniList GraphQL errors (the `errors` array of a response body) --------


class AniListError(_ApiModel):
    """One entry of an AniList GraphQL `errors` array, validated at the boundary.

    AniList follows the GraphQL error shape and adds a numeric `status` (an
    HTTP-style code, e.g. `429` when soft-throttling). Only `status` and
    `message` drive the retry decision, so those are the fields modeled here.
    An entry with junk in either field fails validation and is dropped by
    `_parse_errors` (worst case a soft-throttle reads as non-retryable).
    """

    message: str = ""
    status: int | None = None


# --- AniList Media node (cached GraphQL `Media` record) --------------------


class AniListMediaNode(_ApiModel):
    """One AniList `Media` node, validated once at the cache read boundary.

    Every field defaults to None and an EMPTY DICT must validate to the
    all-None miss node (the `{"data": {"Media": null}}` miss shape reduces to
    `{}` before parsing). The nested `title`/`coverImage` reads are
    `AliasPath`s, which yield the default through a null/absent intermediate
    - the model equivalent of the former chained `.get(...) or {}` walks.
    """

    id: int | None = None
    title_english: str | None = Field(default=None, validation_alias=AliasPath("title", "english"))
    title_romaji: str | None = Field(default=None, validation_alias=AliasPath("title", "romaji"))
    episodes: int | None = None
    cover_image: str | None = Field(default=None, validation_alias=AliasPath("coverImage", "large"))
    banner_image: str | None = Field(default=None, validation_alias="bannerImage")
    format: str | None = None


# --- Sonarr manual-import (candidate read views + outgoing file payload) -----
#
# Derived from the Sonarr v3 OpenAPI `ManualImportResource` (and its nested
# `QualityModel` / `Quality` / `Revision` / `Language` /
# `ImportRejectionResource`), captured in `schemas/sonarr.schema` in the
# source repository. Nullability mirrors the schema exactly (a schema
# `string | null` field -> `str | None`).
#
# Two kinds of model live here:
#   * `Quality` / `Revision` / `QualityModel` are `_WireModel`s
#     (extra="allow"): a candidate's in-context `QualityModel` is read for its
#     axes AND re-emitted verbatim into the outgoing payload, so unknown keys at
#     BOTH nesting levels must survive the round-trip. Every field defaults so
#     the helpers can build/read *partial* objects (a model carrying only
#     `quality`, a quality with just `id`/`name`).
#   * `ManualImportCandidate` / `ImportRejection` / `Language` /
#     `QualityDefinition` are `_ApiModel` reads into the import decision
#     (unknown keys ignored); a resolved `Language` is also re-built fresh
#     with both fields set and POSTed in the file payload.


class QualitySource(StrEnum):
    """Sonarr's `QualitySource` enum (schema `QualitySource`).

    The structured `source` axis of a `Quality`, modeled verbatim from
    the Sonarr OpenAPI schema (`schemas/sonarr.schema` in the source
    repository) - the values are camelCase strings as Sonarr serializes them. Quality is matched on the
    `(source, resolution)` pair (never on the display name), so this enum is the
    authoritative source vocabulary the manual-import quality decision works in.
    `BLURAY_RAW` is a BD remux; `TELEVISION_RAW` is Raw-HD.
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
        string, or `"unknown"` - i.e. None means "no authoritative source", so
        the caller's next precedence layer (our parse, then the configured
        default) gets a chance to fill the axis.

        Args:
            value: A raw `QualitySource` string from Sonarr JSON.

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


class Quality(_WireModel):
    """The nested `quality` object of a Sonarr `QualityModel`.

    Schema `Quality`: `id`/`resolution` are non-null ints, `name` is
    `string | null`, `source` is the `QualitySource` enum (a string).
    Every field defaults because the helpers build and read partial qualities
    (the resolver re-emits only what a definition carries); unknown keys
    survive to the wire (extra="allow").
    """

    id: int | None = None
    name: str | None = None
    source: str | None = None
    resolution: int | None = None


class Revision(_WireModel):
    """The nested `revision` object of a Sonarr `QualityModel` (schema `Revision`)."""

    version: int | None = None
    real: int | None = None
    isRepack: bool | None = None


class QualityModel(_WireModel):
    """A Sonarr `QualityModel` (schema): `{quality, revision}`.

    Used two ways on the manual-import path: a candidate's in-context model is
    read for its structured `quality.source`/`quality.resolution` axes and,
    when no definition matches the resolved quality, re-emitted verbatim into
    the outgoing file payload (unknown keys included). An empty/null incoming
    `quality` folds to None, so "the candidate carries no real quality"
    remains ONE explicit None test everywhere (the historical dict-falsiness).
    """

    quality: Annotated[Quality | None, BeforeValidator(_none_if_falsy)] = None
    revision: Revision | None = None


class Language(_ApiModel):
    """A Sonarr `Language` (schema): `{id, name}`.

    Read+rebuild: `resolve_language_objects` matches these from the
    `/api/v3/language` list and re-builds `{id, name}` fresh - BOTH fields
    explicitly set, so the write dump (`exclude_unset`) always carries them,
    a null `id` included.
    """

    id: int | None = None
    name: str | None = None


class ImportRejection(_ApiModel):
    """One entry of a candidate's `rejections` array (schema `ImportRejectionResource`).

    Only `reason` (`string | null`) is read - it carries the human text the
    sample / already-imported classifier matches against. A junk (non-str,
    non-null) reason fails validation and the entry is skipped, so the
    classifier's `.casefold()` can never crash on a type lie.
    """

    reason: str | None = None


class ManualImportCandidate(_ApiModel):
    """A Sonarr `ManualImportResource`, reduced to the fields planning reads.

    The decision path consults only `path` (the on-disk file to import,
    `string | null` in the schema), `quality` (the in-context
    `QualityModel`, re-emitted verbatim - unknown keys included - when
    it wins the resolution; an empty/null one folds to None) and `rejections`
    (the per-file sample/already-imported flags). `rejections` may be null
    (schema) and, on older Sonarr versions, a bare string per entry rather
    than an `ImportRejectionResource` object; both fold to an
    `ImportRejection`, and non-str/non-dict junk entries are skipped.
    """

    path: str | None = None
    quality: Annotated[QualityModel | None, BeforeValidator(_none_if_falsy)] = None
    rejections: tuple[ImportRejection, ...] = ()

    @field_validator("rejections", mode="before")
    @classmethod
    def _fold_rejections(cls, value: object) -> object:
        """Fold str/dict rejection entries to `ImportRejection`; skip junk."""

        if not isinstance(value, list):
            return ()
        folded: list[ImportRejection] = []
        for rejection in cast("list[object]", value):
            if isinstance(rejection, str):
                folded.append(ImportRejection(reason=rejection))
            elif isinstance(rejection, dict):
                try:
                    folded.append(ImportRejection.model_validate(rejection))
                except ValidationError:
                    continue  # junk reason: skip the entry, keep the candidate
            elif isinstance(rejection, ImportRejection):
                folded.append(rejection)  # field-name construction passes models
        return folded

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(raw)


class ManualImportFile(_WireModel):
    """One outgoing `ManualImport` command file entry (the POST payload).

    Built by the Sonarr strategy from a planned `import` decision and POSTed
    via `model_dump(exclude_unset=True)`. `quality` is the only optional
    field (omitted from the wire, never sent as `None`, when unset - Sonarr
    then falls back to Unknown); the builder always sets it.
    """

    path: str
    seriesId: int
    episodeIds: list[int]
    releaseGroup: str
    downloadId: str
    languages: list[Language]
    quality: QualityModel | None = None


# --- Sonarr queue (`/api/v3/queue` records) -------------------------------
#
# Derived from the Sonarr v3 OpenAPI `QueueResource`, captured in
# `schemas/sonarr.schema` in the source repository. The endpoint pages its
# records under a wrapper object's `records` array.


class QueueRecord(_ApiModel):
    """One Sonarr `QueueResource` record, reduced to the fields the wait reads.

    The wait/import decision consults only `download_id` (the infohash Sonarr
    stores uppercased, matched case-insensitively to pick a torrent's records -
    `string | null` in the schema), `state`
    (`trackedDownloadState`: `downloading` / `importPending` / ...) and
    `status` (`trackedDownloadStatus`: `ok` / `warning` / `error`).
    `state`/`status` are the `string | null` rendering of their schema
    enums. Every field folds junk to None INDEPENDENTLY, so a queue record is
    never dropped over one bad field (a dropped record could route a wait to a
    double-importing step-in).
    """

    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    state: _LenientStr = Field(default=None, validation_alias="trackedDownloadState")
    status: _LenientStr = Field(default=None, validation_alias="trackedDownloadStatus")

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(raw)


# --- Arr history (`/api/v3/history/since` records) -------------------------
#
# Derived from the Sonarr/Radarr v3 OpenAPI `HistoryResource`. The endpoint
# returns a bare, date-ascending array; `id` is the per-arr autoincrement, so
# it doubles as a monotone cursor.


class HistoryRecord(_ApiModel):
    """One arr `HistoryResource` record, reduced to what the activity scan reads.

    `id` is the monotone cursor; `date` the raw ISO8601 arr-clock stamp;
    `item_id` the `seriesId`/`movieId` (0 when absent - no record carries
    both, so one `AliasChoices` serves both arrs); `download_id` the
    infohash (`string | null`; Sonarr uppercases, so compare casefolded);
    `event_type` the camelCase event name; `reason` the `data` map's
    reason value (key read case-insensitively by the before-validator - an
    alias cannot do case-insensitivity). Every field folds junk INDEPENDENTLY,
    so a record is never dropped: a dropped record would be a missed dirty-mark
    and a lagging checkpoint. A junk item id folds to 0 and the record is KEPT
    (arr_activity's `item_id <= 0` drop applies downstream).
    """

    id: _ZeroInt = 0
    date: Annotated[str, BeforeValidator(_stringified)] = ""
    item_id: _ZeroInt = Field(default=0, validation_alias=AliasChoices("seriesId", "movieId"))
    event_type: _BlankStr = Field(default="", validation_alias="eventType")
    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    reason: _LenientStr = None

    @model_validator(mode="before")
    @classmethod
    def _lift_reason(cls, data: object) -> object:
        """Lift the `data` map's reason value, matching its key case-insensitively."""

        if not isinstance(data, dict):
            return data
        record = cast("dict[str, Any]", data)
        raw_data = record.get("data")
        if "reason" in record or not isinstance(raw_data, dict):
            return record
        for key, value in cast("dict[str, Any]", raw_data).items():
            if key.casefold() == "reason" and isinstance(value, str):
                return {**record, "reason": value}
        return record


# --- Sonarr quality definitions (`/api/v3/qualitydefinition`) --------------


class QualityDefinition(_ApiModel):
    """One Sonarr `QualityDefinitionResource` (schema), reduced to `quality`.

    Read-and-re-emit: `resolve_quality` matches a definition by its nested
    `quality.source`/`quality.resolution` pair and re-emits the matched
    `Quality` verbatim (a `_WireModel`, so its unknown keys survive)
    into the outgoing `QualityModel`. Only the nested `quality` is
    consumed; an empty/null one folds to None so the resolver's skip stays one
    explicit None test.
    """

    quality: Annotated[Quality | None, BeforeValidator(_none_if_falsy)] = None


# --- Sonarr commands (`/api/v3/command`) -----------------------------------


class CommandBody(_WireModel):
    """One outgoing `/api/v3/command` POST body (a Sonarr command request).

    Constructed by the strategy and POSTed via `model_dump(exclude_unset=True)`.
    `name` is the command name (always sent); `importMode` / `files` are
    the extra fields the `ManualImport` command carries - unset (and so
    omitted from the wire) for `RefreshMonitoredDownloads`, which sends only
    `{"name"}`.
    """

    name: str
    importMode: str | None = None
    files: list[ManualImportFile] | None = None


def _int_entries(value: object) -> object:
    """Fold an episode-id array: keep the int entries, fold junk/None to ()."""

    if isinstance(value, list):
        return [i for i in cast("list[object]", value) if isinstance(i, int)]
    return ()


class CommandFile(_ApiModel):
    """One file of a `ManualImport` command's `body.files[]` (read back).

    Surfaced from the `/api/v3/command` list so the in-flight guard can tell
    whether an accepted-but-still-running ManualImport already covers a download:
    `download_id` is the primary match key (the infohash a queue-driven import
    carries, `string | null` in the schema - absent for a folder/season-pack
    import), with `path` and `episode_ids` as the fallback signals. Every
    field folds junk independently, so a dict entry never fails validation.
    """

    path: _LenientStr = None
    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    series_id: _ZeroInt = Field(default=0, validation_alias="seriesId")
    episode_ids: Annotated[tuple[int, ...], BeforeValidator(_int_entries)] = Field(
        default=(),
        validation_alias="episodeIds",
    )


class CommandResource(_ApiModel):
    """A Sonarr `CommandResource` (schema), reduced to the fields read back.

    A command POST returns this with the queued command `id`, and the status
    poll reads `status` (the `CommandStatus` enum: `queued` / `started` /
    `completed` / ...) to know when a rescan has settled. `id` is a non-null
    schema int (`0` when absent so the caller drops it); `status` /
    `result` are the `string | null` rendering of their schema enums.

    The `/api/v3/command` LIST poll reads the extra fields for the in-flight
    ManualImport guard: `name` (the command name, e.g. `ManualImport`),
    `message` (the progress text, e.g. `"Processing file 4 of 8"` /
    `"Manually imported 10 files"` - kept for a later wait-view enrichment) and
    `files` (the per-file rows from `body.files`, each a `CommandFile`
    carrying the download id / path / episode ids that say which download a
    still-running import covers). All default to empty so the POST/status callers
    that only read `id`/`status`/`result` are unaffected. Every field folds
    junk independently and a junk `files[]` entry is skipped WITHOUT dropping
    the command - a dropped CommandResource would blind the in-flight guard
    (the double-import direction).
    """

    id: _ZeroInt = 0
    status: _LenientStr = None
    result: _LenientStr = None
    name: _LenientStr = None
    message: _LenientStr = None
    # `files` lives under the nested `body` object (the original command
    # request Sonarr echoes back); the POST/status responses omit it.
    files: tuple[CommandFile, ...] = Field(default=(), validation_alias=AliasPath("body", "files"))

    @field_validator("files", mode="before")
    @classmethod
    def _lenient_files(cls, value: object) -> object:
        """Skip junk `files[]` entries; never fail the whole command over one."""

        if not isinstance(value, list):
            return ()
        kept: list[CommandFile] = []
        for entry in cast("list[object]", value):
            try:
                kept.append(CommandFile.model_validate(entry))
            except ValidationError:
                continue
        return kept

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(raw)


# --- Radarr movie files (`/api/v3/moviefile` records) ----------------------
#
# Derived from the Radarr v3 OpenAPI `MovieFileResource`, captured in
# `schemas/radarr.schema` in the source repository. Nullability mirrors the
# schema exactly (a schema `string | null` field -> `str | None`).


class MovieFile(_ApiModel):
    """A Radarr `MovieFileResource`, reduced to the fields the syncer reads.

    `get_radarr_release_dict` reads each movie file into the shared
    `ArrReleaseDict` decision (release group -> existing-file sizes), so a
    movie file is READ into a decision, not re-emitted: it follows the
    `SonarrEpisode` precedent (a fail-open list read - a record with
    junk in a typed field is skipped with a warning). Only `release_group`
    (`string | null` in the schema) and `size` (a non-null `int64`) are
    consumed. Validated at the client boundary.
    """

    release_group: str | None = Field(default=None, validation_alias="releaseGroup")
    size: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(raw)


# --- Sonarr parse (`/api/v3/parse` `parsedEpisodeInfo`) -------------------


def _tuple_or_empty(value: object) -> object:
    """Fold a null/absent number array to () (Sonarr nulls empty arrays)."""

    return value or ()


class ParsedFileInfo(_ApiModel):
    """Sonarr's series-AGNOSTIC parse of one filename, narrowed to what assignment reads.

    Validated from a raw `/api/v3/parse` response body: every field reads
    through an `AliasPath` into the nested `parsedEpisodeInfo` object - NOT
    the response's `episodes` array (Sonarr's *series-matched* episodes,
    which is empty whenever the release title can't be matched to a series in
    the library). `episode_numbers` (paired with `season_number`) drives
    the exact `(season, episode)` assignment; `absolute_episode_numbers`
    drives the absolute-index fallback. Both are read straight from the
    release name, so they are populated even when Sonarr can't match the title
    to a series - which is exactly the case (specials, alias titles) the old
    series-matched parse failed.

    `season_number` is whatever Sonarr reported and is meaningful only when
    `episode_numbers` is non-empty (an absolute-numbered name reports season 0).
    """

    season_number: int | None = Field(
        default=None,
        validation_alias=AliasPath("parsedEpisodeInfo", "seasonNumber"),
    )
    episode_numbers: Annotated[tuple[int, ...], BeforeValidator(_tuple_or_empty)] = Field(
        default=(),
        validation_alias=AliasPath("parsedEpisodeInfo", "episodeNumbers"),
    )
    absolute_episode_numbers: Annotated[tuple[int, ...], BeforeValidator(_tuple_or_empty)] = Field(
        default=(),
        validation_alias=AliasPath("parsedEpisodeInfo", "absoluteEpisodeNumbers"),
    )
    special: Annotated[bool, BeforeValidator(bool)] = Field(
        default=False,
        validation_alias=AliasPath("parsedEpisodeInfo", "special"),
    )

    @classmethod
    def from_parse_resource(cls, body: dict[str, Any]) -> Self:
        """Thin shim over `model_validate` (the historical parse entrypoint)."""

        return cls.model_validate(body)
