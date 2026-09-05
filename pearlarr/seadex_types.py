"""Domain vocabulary (the `seadex_dict` records) plus the pydantic models arr/AniList JSON validates into."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    NamedTuple,
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
    """One parsed `{season, episode, size}` record for a SeaDex file."""

    season: int | None = None
    episode: int | None = None
    size: int = 0


@dataclass
class SeadexUrlItem:
    """One SeaDex url record within a release group."""

    url: str = ""
    files: list[str] = field(default_factory=list[str])
    size: list[int] = field(default_factory=list[int])
    tracker: Tracker = Tracker.OTHER
    """A SeaDex `Tracker` object (not a str). The notifier renders it as the link text of a grab embed."""
    is_public: bool = True
    is_dual_audio: bool = False
    infohash: str | None = None
    download: bool = False
    is_fallback: bool = False
    """A public alternative added because the preferred release is private-only."""
    upgrade: bool = False
    """A size upgrade over a copy the Arr already holds. Never set without `download`."""
    episodes: list[EpisodeRecord] = field(default_factory=list[EpisodeRecord])

    def __post_init__(self) -> None:
        # Blank -> None: an empty `hashes` filter matches every torrent in the qbit dedup, and "" collides
        # with the cache's _NO_HASH.
        if self.infohash is not None:
            self.infohash = self.infohash.strip() or None

    def flag(self, *, upgrade: bool = False) -> None:
        """Mark the url to grab, `upgrade` also marking a size upgrade. Never clears an upgrade already set."""

        self.download = True
        if upgrade:
            self.upgrade = True

    def unflag(self) -> None:
        """Clear the grab. `upgrade` clears with `download`: it describes the grab, so it must never outlive it."""

        self.download = False
        self.upgrade = False


@dataclass
class SeadexReleaseGroupItem:
    """One SeaDex release-group record, keyed by url under `urls`."""

    urls: dict[str, SeadexUrlItem] = field(default_factory=dict[str, SeadexUrlItem])
    tags: frozenset[Tag] = field(default_factory=frozenset[Tag])
    all_episodes: list[EpisodeRecord] | None = None
    """`None` until the episode parser has run, distinct from an empty list (it ran and found nothing)."""


SeadexDict = dict[str, SeadexReleaseGroupItem]
"""SeaDex release groups keyed by group name."""


def flagged_urls(seadex_dict: SeadexDict) -> list[tuple[str, SeadexUrlItem, str]]:
    """The urls flagged to grab that carry an infohash, as `(group, url item, infohash)` triples.

    Narrow at the riders, never here.
    """

    return [
        (srg, url_item, url_item.infohash)
        for srg, srg_item in seadex_dict.items()
        for url_item in srg_item.urls.values()
        if url_item.download and url_item.infohash
    ]


# Folded into the config's selection digest: bump when the release-selection
# rules change in code so every cached verdict re-checks once after an upgrade.
SELECTION_RULES_VERSION: int = 1

SONARR_MISSING_KEY: int = 999
"""Out-of-range stand-in for a missing Sonarr `seasonNumber`/`episodeNumber`, never colliding with a real one."""


class EpisodeKey(NamedTuple):
    """The folded `(season, episode)` index key, always concrete ints."""

    season: int
    episode: int


def season_episode_key(season: int | None, episode: int | None) -> EpisodeKey:
    """The `EpisodeKey` for a possibly-missing pair, collapsing to the sentinel."""

    return EpisodeKey(
        season if season is not None else SONARR_MISSING_KEY,
        episode if episode is not None else SONARR_MISSING_KEY,
    )


# --- shared plumbing ----------------------------------------------------------

# (connect, read) seconds, shared by the arr http client and the qBittorrent adapter.
# A hung service then surfaces as a transient miss instead of blocking the run.
ARR_REQUEST_TIMEOUT_S = (5, 30)

# The recursive JSON value shape.
type Json = bool | int | float | str | Sequence["Json"] | Mapping[str, "Json"] | None


def coerce_int(value: object) -> int | None:
    """Best-effort int, or None for a non-numeric / NaN value."""

    if isinstance(value, bool):
        return int(value)
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


class _ApiModel(BaseModel):
    """Frozen boundary read model: unknown keys ignored, field-name kwargs allowed.

    `validate_by_name` is required, or constructing an aliased field by field name no-ops to the default.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", validate_by_name=True)


class _WireModel(BaseModel):
    """Frozen wire re-emit shape: unknown keys VALIDATE and RE-EMIT (extra="allow").

    Write dumps are `model_dump(exclude_unset=True)`, NEVER `exclude_none`: an explicitly-set None must reach
    the wire, so set every key the body needs.
    """

    model_config = ConfigDict(frozen=True, extra="allow")


class BoundaryContractError(RuntimeError):
    """A strict library read got a non-empty payload with zero valid records."""


def validation_summary(e: ValidationError) -> str:
    """A log-safe summary of a validation failure: field locs + error types only.

    Never `str(e)`, which embeds the raw input values.
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
    """Validate each raw record into `model`, warning on and skipping the ones that fail.

    With `strict=True` a non-empty `raw` that validates to NOTHING raises `BoundaryContractError`.
    """

    validated: list[ModelT] = []
    for index, record in enumerate(raw):
        try:
            validated.append(model.model_validate(record))
        except ValidationError as e:
            # Deferred: a top-level .output import cycles back here through manual_import.
            # Skip-arm only, so the all-valid hot path never touches the import machinery.
            from .output import hub_warn

            hub_warn(f"Skipping malformed {model.__name__} record [{index}] ({validation_summary(e)})")
    if strict and raw and not validated:
        msg = f"none of the {len(raw)} {model.__name__} records validated - refusing to treat it as empty"
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


def _lax_bool(value: object) -> bool:
    """Per-field lenient fold: real bools and recognized spellings parse, junk folds to False."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in ("true", "1")
    return False


# Reusable lenient field shapes.
type _LenientStr = Annotated[str | None, BeforeValidator(_str_or_none)]
type _BlankStr = Annotated[str, BeforeValidator(_str_or_blank)]
type _ZeroInt = Annotated[int, BeforeValidator(_int_or_zero)]
type _LaxBool = Annotated[bool, BeforeValidator(_lax_bool)]


def _validate_skipping_junk[ModelT: _ApiModel](model: type[ModelT], value: object) -> object:
    """Nested-array before-validator: validate each entry into `model`, skipping junk without failing the array."""

    if not isinstance(value, list):
        return ()
    kept: list[ModelT] = []
    for entry in cast("list[object]", value):
        try:
            kept.append(model.model_validate(entry))
        except ValidationError:
            continue
    return kept


# --- shared progress sink ----------------------------------------------------


class ProgressSink(Protocol):
    """Sink for step progress (`fraction` is 0-1), driving the boot cockpit's live bar.

    Protocol, not an ABC (house rule), so the data and gateway modules need not import the output layer.
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...


# --- Arr items (Sonarr series / Radarr movies) ------------------------------


@runtime_checkable
class ArrItem(Protocol):
    """The attribute surface shared by a Sonarr series and a Radarr movie.

    Protocol, not an ABC (house-rule exception): callers `isinstance`-check it, and an ABC would silently
    flip those checks to False.
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
    """An `ArrItem` keyed on `tvdbId`."""

    @property
    def tvdbId(self) -> int: ...


@runtime_checkable
class RadarrItem(ArrItem, Protocol):
    """An `ArrItem` keyed on `tmdbId`."""

    @property
    def tmdbId(self) -> int: ...


class SonarrSeries(_ApiModel):
    """One Sonarr `/api/v3/series` record, narrowed to the `SonarrItem` surface.

    camelCase on purpose: the fields satisfy the protocol directly and `IdField.item_attr` reads them by name.
    """

    id: int = 0
    title: str = ""
    monitored: bool = True
    tvdbId: int = 0
    imdbId: str | None = None


class RadarrMovie(_ApiModel):
    """One Radarr `/api/v3/movie` record, narrowed to the `RadarrItem` surface (camelCase as in `SonarrSeries`)."""

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
    """One Sonarr `/api/v3/episode` record."""

    id: int = 0
    season_number: int | None = Field(default=None, validation_alias="seasonNumber")
    episode_number: int | None = Field(default=None, validation_alias="episodeNumber")
    episode_file_id: int = Field(default=0, validation_alias="episodeFileId")
    monitored: bool = True
    episode_file: Annotated[SonarrEpisodeFile | None, BeforeValidator(_none_if_falsy)] = Field(
        default=None,
        validation_alias="episodeFile",
    )
    """An empty/null `episodeFile` folds to `None`."""


def index_episodes_by_key(ep_list: Iterable[SonarrEpisode]) -> dict[EpisodeKey, SonarrEpisode]:
    """Index Sonarr episodes by `season_episode_key`, the first record winning."""

    index: dict[EpisodeKey, SonarrEpisode] = {}
    for ep in ep_list:
        index.setdefault(season_episode_key(ep.season_number, ep.episode_number), ep)
    return index


@dataclass(frozen=True, slots=True)
class ArrReleases:
    """The Arr's existing files for one entry, folded by release group.

    Keys hold the tag as the arr wrote it, and only an EMPTY tag makes a file untagged.
    """

    tagged: Mapping[str, tuple[int, ...]] = field(default_factory=dict[str, tuple[int, ...]])
    """Each tagged release group's existing-file sizes, insertion-ordered."""

    untagged: tuple[int, ...] = ()
    """Sizes of files with no release group (Radarr only)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "tagged", MappingProxyType(dict(self.tagged)))

    def __hash__(self) -> int:
        # The generated hash would reject the Mapping field. Eq ignores key order.
        return hash((frozenset(self.tagged.items()), self.untagged))

    @classmethod
    def from_files(
        cls,
        files: "Iterable[MovieFile | SonarrEpisodeFile]",
        *,
        keep_untagged: bool,
    ) -> Self:
        """Fold arr file records into one per-entry record.

        Sonarr passes `keep_untagged=False`: its untagged files ride the episode list and would double-count.
        """

        tagged: dict[str, list[int]] = {}
        untagged: list[int] = []
        for arr_file in files:
            if arr_file.release_group:
                sizes = tagged.setdefault(arr_file.release_group, [])
                if arr_file.size is not None:
                    sizes.append(arr_file.size)
            elif keep_untagged:
                untagged.append(arr_file.size or 0)
        return cls(
            tagged={rg: tuple(sizes) for rg, sizes in tagged.items()},
            untagged=tuple(untagged),
        )

    def group_count(self) -> int:
        """How many entries `groups_label` renders: tagged names, plus one for untagged files."""

        return len(self.tagged) + (1 if self.untagged else 0)

    def groups_label(self) -> str:
        """The group names comma-joined for a log line: `(none)` marks untagged files, `(no files)` an empty record."""

        return ", ".join([*self.tagged, *(["(none)"] if self.untagged else [])]) or "(no files)"

    def replaced_groups(self) -> tuple[str, ...]:
        """Every tagged name a grab would replace (the notify `Replacing` field)."""

        return tuple(self.tagged)


type TvdbMappings = dict[int, list[tuple[int, int | None]]]
"""AniBridge TVDB season -> inclusive `(start, end)` episode ranges."""


# --- AniList GraphQL errors (the `errors` array of a response body) --------


class AniListError(_ApiModel):
    """One entry of an AniList GraphQL `errors` array.

    `status` is an HTTP-style code (`429` when soft-throttling).
    """

    message: str = ""
    status: int | None = None


# --- AniList Media node (cached GraphQL `Media` record) --------------------


class AniListMediaNode(_ApiModel):
    """One AniList `Media` node, validated at the cache read boundary.

    An EMPTY DICT must validate to the all-None miss node (`{"data": {"Media": null}}` reduces to `{}` first).
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
# Derived from the Sonarr v3 OpenAPI `ManualImportResource` (`schemas/sonarr.schema`), nullability mirroring
# the schema exactly. `Quality`, `Revision` and `QualityModel` are `_WireModel`s: a candidate's in-context
# `QualityModel` is read for its axes AND re-emitted verbatim, so unknown keys at BOTH nesting levels must
# survive the round-trip.


class QualitySource(StrEnum):
    """Sonarr's `QualitySource` enum, the structured `source` axis of a `Quality`.

    Quality is matched on the `(source, resolution)` pair, NEVER on the display name.
    `BLURAY_RAW` is a BD remux, `TELEVISION_RAW` is Raw-HD.
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
        """A real source from a raw enum string (case-insensitive), or None when undetermined."""

        return _SOURCE_BY_FOLDED.get(value.casefold()) if value else None


# Case-folded value to member. UNKNOWN is excluded so it folds to None.
_SOURCE_BY_FOLDED: dict[str, QualitySource] = {
    m.value.casefold(): m for m in QualitySource if m is not QualitySource.UNKNOWN
}


class Quality(_WireModel):
    """The nested `quality` object of a Sonarr `QualityModel` (schema `Quality`)."""

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
    """A Sonarr `QualityModel` (schema): `{quality, revision}`."""

    quality: Annotated[Quality | None, BeforeValidator(_none_if_falsy)] = None
    revision: Revision | None = None


class Language(_ApiModel):
    """A Sonarr `Language` (schema): `{id, name}`.

    Rebuilt with BOTH fields explicitly set, so the `exclude_unset` write dump carries them, null `id` included.
    """

    id: int | None = None
    name: str | None = None


class ImportRejection(_ApiModel):
    """One entry of a candidate's `rejections` array (schema `ImportRejectionResource`)."""

    reason: str | None = None
    """The human text the sample / already-imported classifier matches against (`string | null` in the schema)."""


class ManualImportCandidate(_ApiModel):
    """A Sonarr `ManualImportResource`, reduced to the fields planning reads."""

    path: str | None = None
    """The on-disk file to import (`string | null` in the schema)."""
    quality: Annotated[QualityModel | None, BeforeValidator(_none_if_falsy)] = None
    """The in-context `QualityModel`, re-emitted verbatim (unknown keys included). An empty/null one folds to None."""
    rejections: tuple[ImportRejection, ...] = ()
    """May be null and, on older Sonarr versions, a bare string per entry rather than an object."""

    @field_validator("rejections", mode="before")
    @classmethod
    def _fold_rejections(cls, value: object) -> object:
        """Fold str/dict rejection entries to `ImportRejection`, skipping junk."""

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


class ManualImportFile(_WireModel):
    """One outgoing `ManualImport` command file entry, POSTed via `model_dump(exclude_unset=True)`."""

    path: str
    seriesId: int
    episodeIds: list[int]
    releaseGroup: str
    downloadId: str | None = None
    """UNSET for a dead-tracked folder-mode entry: a downloadId re-enters Sonarr's poisoned tracked branch."""
    languages: list[Language]
    quality: QualityModel | None = None
    """Unset stays off the wire, never sent as `None`, and Sonarr falls back to Unknown."""


# --- Sonarr queue (`/api/v3/queue` records) -------------------------------
#
# Derived from the Sonarr v3 OpenAPI `QueueResource`. The endpoint pages its
# records under a wrapper object's `records` array.


class QueueRecord(_ApiModel):
    """One Sonarr `QueueResource` record, reduced to the fields the wait reads."""

    id: _ZeroInt = 0
    """`queue_delete`'s handle: any one of a download's rows dismisses the whole download. 0 means unusable."""
    series_id: _ZeroInt = Field(default=0, validation_alias="seriesId")
    """0 when Sonarr never matched a series. The queue close skips those rows: Sonarr's dismissal 500s on them."""
    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    """The infohash. Sonarr stores it uppercased, so match case-insensitively."""
    state: _LenientStr = Field(default=None, validation_alias="trackedDownloadState")
    """`downloading`, `importPending`, and so on."""
    status: _LenientStr = Field(default=None, validation_alias="trackedDownloadStatus")
    """`ok`, `warning`, or `error`."""


# --- Arr history (`/api/v3/history/since` records) -------------------------
#
# Derived from the Sonarr/Radarr v3 OpenAPI `HistoryResource`. The endpoint
# returns a bare, date-ascending array.


class HistoryRecord(_ApiModel):
    """One arr `HistoryResource` record, reduced to what the activity scan reads."""

    id: _ZeroInt = 0
    """The per-arr autoincrement, doubling as the monotone cursor."""
    date: Annotated[str, BeforeValidator(_stringified)] = ""
    """The raw ISO8601 arr-clock stamp."""
    item_id: _ZeroInt = Field(default=0, validation_alias=AliasChoices("seriesId", "movieId"))
    """The `seriesId` or `movieId`. No record carries both, so one `AliasChoices` serves both arrs."""
    event_type: _BlankStr = Field(default="", validation_alias="eventType")
    """The camelCase event name."""
    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    """The infohash, uppercased by Sonarr, so compare casefolded."""
    reason: _LenientStr = None
    """The `data` map's reason value, its key read case-insensitively (an alias cannot)."""

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


class HistoryPage(_ApiModel):
    """One `/api/v3/history` page envelope, reduced to its `records` array."""

    records: tuple[HistoryRecord, ...] = ()

    @field_validator("records", mode="before")
    @classmethod
    def _lenient_records(cls, value: object) -> object:
        """Skip junk `records[]` entries, never failing the whole page over one."""

        return _validate_skipping_junk(HistoryRecord, value)


# --- Arr download clients (`/api/v3/downloadclient`) ------------------------


def _priority_or_lowest(value: object) -> int:
    """Keep an int priority, folding junk to 50 (the arrs' lowest)."""

    return value if isinstance(value, int) and not isinstance(value, bool) else 50


class DownloadClientField(_ApiModel):
    """One `{name, value}` settings field of a download-client definition."""

    name: _LenientStr = None
    value: _LenientStr = None


class DownloadClientRecord(_ApiModel):
    """One arr `DownloadClientResource`, reduced to the category-fallback read.

    `implementation` names the client type (`QBittorrent`). `priority` runs 1 (highest, the arr default) to 50.
    """

    enable: _LaxBool = False
    implementation: _LenientStr = None
    priority: Annotated[int, BeforeValidator(_priority_or_lowest)] = 50
    fields: tuple[DownloadClientField, ...] = ()

    @field_validator("fields", mode="before")
    @classmethod
    def _lenient_fields(cls, value: object) -> object:
        """Skip junk `fields[]` entries, never failing the definition over one."""

        return _validate_skipping_junk(DownloadClientField, value)

    def field_value(self, name: str) -> str | None:
        """The named settings field's value, or None when absent or blank."""

        return next((field.value or None for field in self.fields if field.name == name), None)


# --- Sonarr remote path mappings (`/api/v3/remotepathmapping`) --------------


class RemotePathMapping(_ApiModel):
    """One Sonarr `RemotePathMappingResource`: a download-client path mapped to Sonarr's view.

    `host` is the client host as configured IN SONARR, so it only ever tiebreaks, never excludes.
    """

    host: _LenientStr = None
    remote_path: _LenientStr = Field(default=None, validation_alias="remotePath")
    local_path: _LenientStr = Field(default=None, validation_alias="localPath")


# --- Sonarr download client config (`/api/v3/config/downloadclient`) ----------


class DownloadClientConfig(_ApiModel):
    """Sonarr's `DownloadClientConfigResource`, reduced to the completed-download-handling switch.

    Off, Sonarr parks a clean `importPending` download forever and Pearlarr imports it. Defaults to on, so a
    fail-open read defers rather than racing Sonarr.
    """

    enable_completed_download_handling: bool = Field(default=True, validation_alias="enableCompletedDownloadHandling")


# --- Sonarr quality definitions (`/api/v3/qualitydefinition`) --------------


class QualityDefinition(_ApiModel):
    """One Sonarr `QualityDefinitionResource` (schema), reduced to `quality`."""

    quality: Annotated[Quality | None, BeforeValidator(_none_if_falsy)] = None


# --- Sonarr commands (`/api/v3/command`) -----------------------------------


class CommandBody(_WireModel):
    """One outgoing `/api/v3/command` POST body, dumped with `exclude_unset=True`.

    `importMode` and `files` stay unset (off the wire) for `RefreshMonitoredDownloads`, which sends `{"name"}`.
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
    """One file of a `ManualImport` command's `body.files[]` (read back)."""

    path: _LenientStr = None
    """Fallback match signal, with `episode_ids`."""
    download_id: _LenientStr = Field(default=None, validation_alias="downloadId")
    """The primary match key, absent for a folder or season-pack import."""
    series_id: _ZeroInt = Field(default=0, validation_alias="seriesId")
    episode_ids: Annotated[tuple[int, ...], BeforeValidator(_int_entries)] = Field(
        default=(),
        validation_alias="episodeIds",
    )
    """Fallback match signal, with `path`."""


class CommandResource(_ApiModel):
    """A Sonarr `CommandResource` (schema), reduced to the fields read back."""

    id: _ZeroInt = 0
    """`0` when absent, so the caller drops it."""
    status: _LenientStr = None
    """The `CommandStatus` enum (`queued`, `started`, `completed`, and so on)."""
    result: _LenientStr = None
    """The `string | null` rendering of its schema enum."""
    name: _LenientStr = None
    """The command name, e.g. `ManualImport`."""
    message: _LenientStr = None
    """The progress text, e.g. `"Processing file 4 of 8"`."""
    files: tuple[CommandFile, ...] = Field(default=(), validation_alias=AliasPath("body", "files"))
    """The rows of the nested `body` Sonarr echoes back. The POST and status responses omit it."""

    @field_validator("files", mode="before")
    @classmethod
    def _lenient_files(cls, value: object) -> object:
        """Skip junk `files[]` entries, never failing the whole command over one."""

        return _validate_skipping_junk(CommandFile, value)


# --- Radarr movie files (`/api/v3/moviefile` records) ----------------------


class MovieFile(_ApiModel):
    """A Radarr `MovieFileResource`, reduced to the fields the syncer reads."""

    release_group: str | None = Field(default=None, validation_alias="releaseGroup")
    size: int | None = None


# --- Sonarr parse (`/api/v3/parse` `parsedEpisodeInfo`) -------------------


class ParsedEpisode(NamedTuple):
    """One Sonarr `/parse` series-MATCHED `(season, episode)` pair.

    Persisted as a `{"season", "episode"}` JSON object at the parse-cache seam.
    """

    season: int
    episode: int


class SonarrParse(NamedTuple):
    """One Sonarr `/parse` result: the matched pairs plus the parse-level flag.

    `full_season` (a bare "S0X" name) is parse-level, not per pair. A failed parse stays `None`, never this.
    """

    episodes: list[ParsedEpisode]
    full_season: bool = False


def _tuple_or_empty(value: object) -> object:
    """Fold a null/absent number array to () (Sonarr nulls empty arrays)."""

    return value or ()


class MatchedEpisode(_ApiModel):
    """One series-matched episode from a `/parse` response's `episodes` array."""

    season_number: int = Field(validation_alias="seasonNumber")
    episode_number: int = Field(validation_alias="episodeNumber")
    id: int | None = None
    """Sonarr's episode id, cross-checked against OUR map so a wrong-series title match is refused."""


class ParsedFileInfo(_ApiModel):
    """Sonarr's parse of one filename, narrowed to what assignment reads.

    The `parsedEpisodeInfo` numbers are series-AGNOSTIC (present even with no series match), `matched_episodes`
    is Sonarr's MATCHED resolution. Assignment prefers the agnostic numbers, matched pairs only inside OUR set.
    """

    season_number: int | None = Field(
        default=None,
        validation_alias=AliasPath("parsedEpisodeInfo", "seasonNumber"),
    )
    """Meaningful only when `episode_numbers` is non-empty (an absolute-numbered name reports season 0)."""
    episode_numbers: Annotated[tuple[int, ...], BeforeValidator(_tuple_or_empty)] = Field(
        default=(),
        validation_alias=AliasPath("parsedEpisodeInfo", "episodeNumbers"),
    )
    """Drives the exact `(season, episode)` assignment, paired with `season_number`."""
    absolute_episode_numbers: Annotated[tuple[int, ...], BeforeValidator(_tuple_or_empty)] = Field(
        default=(),
        validation_alias=AliasPath("parsedEpisodeInfo", "absoluteEpisodeNumbers"),
    )
    """Drives the absolute-index fallback."""
    special: _LaxBool = Field(
        default=False,
        validation_alias=AliasPath("parsedEpisodeInfo", "special"),
    )
    full_season: _LaxBool = Field(
        default=False,
        validation_alias=AliasPath("parsedEpisodeInfo", "fullSeason"),
    )
    """A season-pack-shaped name (bare "S01"). Its matched pairs span the season, never one file's claims."""
    offline: bool = False
    """Built by the offline SxxExx fallback, not Sonarr's parser (`ParseResource` has no such property)."""
    matched_episodes: tuple[MatchedEpisode, ...] = Field(default=(), validation_alias="episodes")
    """Sonarr's series-matched pairs. The exact leg's fallback when the name carries no `(season, episode)`."""

    @field_validator("matched_episodes", mode="before")
    @classmethod
    def _lenient_matched(cls, value: object) -> object:
        """Fold a junk `episodes[]`, or ANY junk entry in it, to ().

        All-or-nothing on purpose: dropping one entry would shorten a span and slip the every-pair check.
        """

        # tuple included: direct construction passes the field's own type.
        if not isinstance(value, (list, tuple)):
            return ()
        kept: list[MatchedEpisode] = []
        for entry in cast("Sequence[object]", value):
            try:
                kept.append(MatchedEpisode.model_validate(entry))
            except ValidationError:
                return ()
        return kept
