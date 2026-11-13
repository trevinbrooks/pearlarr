"""The Sonarr and Radarr API models: the records their JSON validates into, the bodies sent, and the item protocols."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Protocol, Self, cast, runtime_checkable

from pydantic import AliasChoices, AliasPath, BeforeValidator, Field, ValidationError, field_validator, model_validator

from .boundary import (
    ApiModel,
    BlankStr,
    LaxBool,
    LenientDay,
    LenientInt,
    LenientStr,
    LenientTime,
    WireModel,
    ZeroInt,
    none_if_falsy,
    stringified,
    validate_all_or_none,
    validate_skipping_junk,
)
from .releases import EpisodeKey, season_episode_key
from .tvdb_orders import AIRED_ORDER

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


class SeasonTypeCount(ApiModel):
    """One `seasonTypes[]` element: a TVDB order a series lists and its episode count.

    Sonarr v5's series record and SkyHook's reply send the same shape.
    """

    type: str
    """The order's type name (`official`, `dvd`, `alternate`, ...)."""
    episode_count: int = Field(validation_alias="episodeCount")


class AlternateTitle(ApiModel):
    """One `alternateTitles[]` element: another title Sonarr knows the series by."""

    title: str


@runtime_checkable
class SonarrItem(ArrItem, Protocol):
    """An `ArrItem` keyed on `tvdbId`, with the TVDB orders Sonarr lists for it and its alternate titles."""

    @property
    def tvdbId(self) -> int: ...

    @property
    def seasonType(self) -> str: ...

    @property
    def seasonTypes(self) -> tuple[SeasonTypeCount, ...]: ...

    @property
    def alternateTitles(self) -> tuple[AlternateTitle, ...]: ...


@runtime_checkable
class RadarrItem(ArrItem, Protocol):
    """An `ArrItem` keyed on `tmdbId`."""

    @property
    def tmdbId(self) -> int: ...


class SonarrSeries(ApiModel):
    """One Sonarr `/api/v3/series` record, narrowed to the `SonarrItem` surface.

    camelCase on purpose: the fields satisfy the protocol directly and `IdField.item_attr` reads them by name.
    """

    id: int = 0
    title: str = ""
    monitored: bool = True
    tvdbId: int = 0
    imdbId: str | None = None
    seasonType: BlankStr = AIRED_ORDER
    """The TVDB order Sonarr numbers the series by. Sonarr v4 sends no field and numbers by the official order. A
    null or junk value folds to "", which names no order, so the series is read without its other orders."""
    seasonTypes: tuple[SeasonTypeCount, ...] = ()
    """The orders TVDB lists for the series, with counts. Sonarr v5 only."""
    alternateTitles: tuple[AlternateTitle, ...] = ()
    """The series' other titles, in Sonarr's order. Some name one season, which this model doesn't keep."""

    @field_validator("seasonTypes", mode="before")
    @classmethod
    def _lenient_season_types(cls, value: object) -> object:
        """Skip junk `seasonTypes[]` entries, never failing the series over one."""

        return validate_skipping_junk(SeasonTypeCount, value)

    @field_validator("alternateTitles", mode="before")
    @classmethod
    def _lenient_alternate_titles(cls, value: object) -> object:
        """Skip junk `alternateTitles[]` entries (a null title among them), never failing the series over one."""

        return validate_skipping_junk(AlternateTitle, value)


class RadarrMovie(ApiModel):
    """One Radarr `/api/v3/movie` record, narrowed to the `RadarrItem` surface (camelCase as in `SonarrSeries`)."""

    id: int = 0
    title: str = ""
    monitored: bool = True
    tmdbId: int = 0
    imdbId: str | None = None


# --- Sonarr episodes (`/api/v3/episode` JSON) -----------------------------


class SonarrEpisodeFile(ApiModel):
    """The `episodeFile` sub-record of a Sonarr episode."""

    release_group: str | None = Field(default=None, validation_alias="releaseGroup")
    size: int | None = None


class SonarrEpisode(ApiModel):
    """One Sonarr `/api/v3/episode` record."""

    id: int = 0
    season_number: int | None = Field(default=None, validation_alias="seasonNumber")
    episode_number: int | None = Field(default=None, validation_alias="episodeNumber")
    absolute_episode_number: int | None = Field(default=None, validation_alias="absoluteEpisodeNumber")
    """The series-wide number TVDB gives the episode (specials interleaved), when it has one."""
    title: str = ""
    tvdb_id: int | None = Field(default=None, validation_alias="tvdbId")
    """TVDB's episode id, the key a TVDB order's rows join on."""
    air_date: LenientDay = Field(default=None, validation_alias="airDate")
    """The day the episode aired, None when Sonarr gives none or a value that isn't a date."""
    runtime: LenientInt = None
    """The episode's length in minutes, None when Sonarr gives none or a value that isn't a number."""
    episode_file_id: int = Field(default=0, validation_alias="episodeFileId")
    monitored: bool = True
    episode_file: Annotated[SonarrEpisodeFile | None, BeforeValidator(none_if_falsy)] = Field(
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
class ArrFiles:
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

    def groups(self) -> tuple[str, ...]:
        """Every tagged name a grab would replace (the notify `Replacing` field)."""

        return tuple(self.tagged)


# --- Sonarr manual-import (candidate read views + outgoing file payload) -----
#
# Derived from the Sonarr v3 OpenAPI `ManualImportResource` (`schemas/sonarr.schema`), nullability mirroring
# the schema exactly. `Quality`, `Revision` and `QualityModel` are `WireModel`s: a candidate's in-context
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


class Quality(WireModel):
    """The nested `quality` object of a Sonarr `QualityModel` (schema `Quality`)."""

    id: int | None = None
    name: str | None = None
    source: str | None = None
    resolution: int | None = None


class Revision(WireModel):
    """The nested `revision` object of a Sonarr `QualityModel` (schema `Revision`)."""

    version: int | None = None
    real: int | None = None
    isRepack: bool | None = None


class QualityModel(WireModel):
    """A Sonarr `QualityModel` (schema): `{quality, revision}`."""

    quality: Annotated[Quality | None, BeforeValidator(none_if_falsy)] = None
    revision: Revision | None = None


class Language(ApiModel):
    """A Sonarr `Language` (schema): `{id, name}`.

    Rebuilt with BOTH fields explicitly set, so the `exclude_unset` write dump carries them, null `id` included.
    """

    id: int | None = None
    name: str | None = None


class ImportRejection(ApiModel):
    """One entry of a candidate's `rejections` array (schema `ImportRejectionResource`)."""

    reason: str | None = None
    """The human text the sample / already-imported classifier matches against (`string | null` in the schema)."""


class ManualImportCandidate(ApiModel):
    """A Sonarr `ManualImportResource`, reduced to the fields planning reads."""

    path: str | None = None
    """The on-disk file to import (`string | null` in the schema)."""
    quality: Annotated[QualityModel | None, BeforeValidator(none_if_falsy)] = None
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


class ManualImportFile(WireModel):
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


class QueueRecord(ApiModel):
    """One Sonarr `QueueResource` record, reduced to the fields the wait reads."""

    id: ZeroInt = 0
    """`queue_delete`'s handle: any one of a download's rows dismisses the whole download. 0 means unusable."""
    series_id: ZeroInt = Field(default=0, validation_alias="seriesId")
    """0 when Sonarr never matched a series. The queue close skips those rows: Sonarr's dismissal 500s on them."""
    download_id: LenientStr = Field(default=None, validation_alias="downloadId")
    """The infohash. Sonarr stores it uppercased, so match case-insensitively."""
    state: LenientStr = Field(default=None, validation_alias="trackedDownloadState")
    """`downloading`, `importPending`, and so on."""
    status: LenientStr = Field(default=None, validation_alias="trackedDownloadStatus")
    """`ok`, `warning`, or `error`."""


# --- Arr history (`/api/v3/history/since` records) -------------------------
#
# Derived from the Sonarr/Radarr v3 OpenAPI `HistoryResource`. The endpoint
# returns a bare, date-ascending array.


class HistoryRecord(ApiModel):
    """One arr `HistoryResource` record, reduced to what the activity scan reads."""

    id: ZeroInt = 0
    """The per-arr autoincrement, doubling as the monotone cursor."""
    date: Annotated[str, BeforeValidator(stringified)] = ""
    """The raw ISO8601 arr-clock stamp."""
    item_id: ZeroInt = Field(default=0, validation_alias=AliasChoices("seriesId", "movieId"))
    """The `seriesId` or `movieId`. No record carries both, so one `AliasChoices` serves both arrs."""
    event_type: BlankStr = Field(default="", validation_alias="eventType")
    """The camelCase event name."""
    download_id: LenientStr = Field(default=None, validation_alias="downloadId")
    """The infohash, uppercased by Sonarr, so compare casefolded."""
    reason: LenientStr = None
    """The `data` map's reason value, its key read case-insensitively (an alias cannot)."""

    @model_validator(mode="before")
    @classmethod
    def _lift_reason(cls, data: object) -> object:
        """Lift the `data` map's reason value, matching its key case-insensitively."""

        return _lift_from_data(data, "reason", "reason")


def _lift_from_data(data: object, key: str, field: str) -> object:
    """Copy one `data` map value onto a top-level field, its key matched casefolded.

    A field already present is left alone.
    """

    if not isinstance(data, dict):
        return data
    record = cast("dict[str, Any]", data)
    raw_data = record.get("data")
    if field in record or not isinstance(raw_data, dict):
        return record
    wanted = key.casefold()
    for name, value in cast("dict[str, Any]", raw_data).items():
        if name.casefold() == wanted and isinstance(value, str):
            return {**record, field: value}
    return record


class SonarrHistoryRecord(HistoryRecord):
    """A Sonarr `HistoryResource` row with the import fields the download probe reads."""

    episode_id: ZeroInt = Field(default=0, validation_alias="episodeId")
    """The episode an import row landed on; 0 on rows that carry none."""
    dropped_path: LenientStr = None
    """The `data` map's `droppedPath`: the file's path in the download folder as Sonarr saw it,
    its key read case-insensitively."""

    @model_validator(mode="before")
    @classmethod
    def _lift_dropped_path(cls, data: object) -> object:
        """Lift the `data` map's `droppedPath` onto `dropped_path`."""

        return _lift_from_data(data, "droppedPath", "dropped_path")


class HistoryPage(ApiModel):
    """One `/api/v3/history` page envelope, reduced to its `records` array and row count."""

    records: tuple[SonarrHistoryRecord, ...] = ()
    total_records: ZeroInt = Field(default=0, validation_alias="totalRecords")
    """The envelope's row count; page 1 alone is fetched, so more than `len(records)` means the page is cut."""

    @field_validator("records", mode="before")
    @classmethod
    def _lenient_records(cls, value: object) -> object:
        """Skip junk `records[]` entries, never failing the whole page over one."""

        return validate_skipping_junk(SonarrHistoryRecord, value)


# --- Arr download clients (`/api/v3/downloadclient`) ------------------------


def _priority_or_lowest(value: object) -> int:
    """Keep an int priority, folding junk to 50 (the arrs' lowest)."""

    return value if isinstance(value, int) and not isinstance(value, bool) else 50


class DownloadClientField(ApiModel):
    """One `{name, value}` settings field of a download-client definition."""

    name: LenientStr = None
    value: LenientStr = None


class DownloadClientRecord(ApiModel):
    """One arr `DownloadClientResource`, reduced to the category-fallback read.

    `implementation` names the client type (`QBittorrent`). `priority` runs 1 (highest, the arr default) to 50.
    """

    enable: LaxBool = False
    implementation: LenientStr = None
    priority: Annotated[int, BeforeValidator(_priority_or_lowest)] = 50
    fields: tuple[DownloadClientField, ...] = ()

    @field_validator("fields", mode="before")
    @classmethod
    def _lenient_fields(cls, value: object) -> object:
        """Skip junk `fields[]` entries, never failing the definition over one."""

        return validate_skipping_junk(DownloadClientField, value)

    def field_value(self, name: str) -> str | None:
        """The named settings field's value, or None when absent or blank (whitespace alone is blank)."""

        value = next((field.value for field in self.fields if field.name == name), None)
        return value if value and value.strip() else None


# --- Sonarr remote path mappings (`/api/v3/remotepathmapping`) --------------


class RemotePathMapping(ApiModel):
    """One Sonarr `RemotePathMappingResource`: a download-client path mapped to Sonarr's view.

    `host` is the client host as configured IN SONARR, so it only ever tiebreaks, never excludes.
    """

    host: LenientStr = None
    remote_path: LenientStr = Field(default=None, validation_alias="remotePath")
    local_path: LenientStr = Field(default=None, validation_alias="localPath")


# --- Sonarr download client config (`/api/v3/config/downloadclient`) ----------


class DownloadClientConfig(ApiModel):
    """Sonarr's `DownloadClientConfigResource`, reduced to the completed-download-handling switch.

    Off, Sonarr parks a clean `importPending` download forever and Pearlarr imports it. Defaults to on, so a
    fail-open read defers rather than racing Sonarr.
    """

    enable_completed_download_handling: bool = Field(default=True, validation_alias="enableCompletedDownloadHandling")


# --- Sonarr quality definitions (`/api/v3/qualitydefinition`) --------------


class QualityDefinition(ApiModel):
    """One Sonarr `QualityDefinitionResource` (schema), reduced to `quality`."""

    quality: Annotated[Quality | None, BeforeValidator(none_if_falsy)] = None


# --- Sonarr commands (`/api/v3/command`) -----------------------------------


class CommandBody(WireModel):
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


class CommandFile(ApiModel):
    """One file of a `ManualImport` command's `body.files[]` (read back)."""

    path: LenientStr = None
    """Fallback match signal, with `episode_ids`."""
    download_id: LenientStr = Field(default=None, validation_alias="downloadId")
    """The primary match key, absent for a folder or season-pack import."""
    series_id: ZeroInt = Field(default=0, validation_alias="seriesId")
    episode_ids: Annotated[tuple[int, ...], BeforeValidator(_int_entries)] = Field(
        default=(),
        validation_alias="episodeIds",
    )
    """Fallback match signal, with `path`."""


class CommandResource(ApiModel):
    """A Sonarr `CommandResource` (schema), reduced to the fields read back."""

    id: ZeroInt = 0
    """`0` when absent, so the caller drops it."""
    status: LenientStr = None
    """The `CommandStatus` enum (`queued`, `started`, `completed`, and so on)."""
    result: LenientStr = None
    """The `string | null` rendering of its schema enum."""
    name: LenientStr = None
    """The command name, e.g. `ManualImport`."""
    message: LenientStr = None
    """The progress text, e.g. `"Processing file 4 of 8"`."""
    started: LenientTime = None
    """When Sonarr began running it. None until it starts."""
    files: tuple[CommandFile, ...] = Field(default=(), validation_alias=AliasPath("body", "files"))
    """The rows of the nested `body` Sonarr echoes back. The POST and status responses omit it."""

    @field_validator("files", mode="before")
    @classmethod
    def _lenient_files(cls, value: object) -> object:
        """Skip junk `files[]` entries, never failing the whole command over one."""

        return validate_skipping_junk(CommandFile, value)


# --- Radarr movie files (`/api/v3/moviefile` records) ----------------------


class MovieFile(ApiModel):
    """A Radarr `MovieFileResource`, reduced to the fields the syncer reads."""

    release_group: str | None = Field(default=None, validation_alias="releaseGroup")
    size: int | None = None


# --- Sonarr parse (`/api/v3/parse` `parsedEpisodeInfo`) -------------------


def _tuple_or_empty(value: object) -> object:
    """Fold a null/absent number array to () (Sonarr nulls empty arrays)."""

    return value or ()


class MatchedEpisode(ApiModel):
    """One series-matched episode from a `/parse` response's `episodes` array."""

    season_number: int = Field(validation_alias="seasonNumber")
    episode_number: int = Field(validation_alias="episodeNumber")
    id: int | None = None
    """Sonarr's episode id, checked against OUR map so a wrong-series title match is refused. None for a pair
    `widen_range_key` added."""


class ParsedFileInfo(ApiModel):
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
    special: LaxBool = Field(
        default=False,
        validation_alias=AliasPath("parsedEpisodeInfo", "special"),
    )
    full_season: LaxBool = Field(
        default=False,
        validation_alias=AliasPath("parsedEpisodeInfo", "fullSeason"),
    )
    """A season-pack-shaped name (bare "S01"). Its matched pairs span the season, never one file's claims."""
    series_title: BlankStr = Field(
        default="",
        validation_alias=AliasPath("parsedEpisodeInfo", "seriesTitle"),
    )
    """The series title Sonarr's parser lifted from the name, whether or not it matched a series. "" when it lifted
    none."""
    matched_episodes: tuple[MatchedEpisode, ...] = Field(default=(), validation_alias="episodes")
    """Sonarr's series-matched pairs. The exact leg's fallback when the name carries no `(season, episode)`."""

    @field_validator("matched_episodes", mode="before")
    @classmethod
    def _lenient_matched(cls, value: object) -> object:
        """Fold a junk `episodes[]`, or ANY junk entry in it, to ().

        All-or-nothing on purpose: dropping one entry would shorten a span and slip the every-pair check.
        """

        return validate_all_or_none(MatchedEpisode, value) or ()

    def with_numbers(
        self, *, episodes: tuple[int, ...], absolutes: tuple[int, ...], matched: tuple[MatchedEpisode, ...]
    ) -> Self:
        """A copy of this parse with new episode numbers, absolute numbers, and matched pairs."""

        return self.model_copy(
            update={"episode_numbers": episodes, "absolute_episode_numbers": absolutes, "matched_episodes": matched}
        )
