"""Sonarr REST client over the raw `/api/v3` endpoints."""

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import cast, override

from pydantic import BaseModel, ConfigDict, ValidationError

from .arr_http import ArrHttp, DeleteOutcome
from .json_narrow import is_json_obj
from .manual_import import PendingImport
from .output import hub_warn
from .seadex_types import (
    CommandBody,
    CommandResource,
    DownloadClientConfig,
    HistoryPage,
    HistoryRecord,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    ParsedEpisode,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    RemotePathMapping,
    SonarrEpisode,
    SonarrItem,
    SonarrParse,
    SonarrSeries,
    validate_each,
    validation_summary,
)

# Per-request timeout (seconds) for the manual-import reads over a slow remote mount: the folder scan and the
# single-file parse. Bounded so a hung read surfaces as a transient miss (retry) instead of blocking the run.
MANUAL_IMPORT_TIMEOUT_S = 120


class _ParsedEpisode(BaseModel):
    """One `ParseResource.episodes[]` entry, reduced to the two numbers read."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    seasonNumber: int | None = None
    episodeNumber: int | None = None


class AbstractSonarrClient(ABC):
    """The Sonarr read/command surface."""

    @abstractmethod
    def all_series(self) -> list[SonarrItem]: ...

    @abstractmethod
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None: ...

    @abstractmethod
    def parse(self, filename: str) -> SonarrParse | None: ...

    @abstractmethod
    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None: ...

    @abstractmethod
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None: ...

    @abstractmethod
    def manual_import_candidates_by_folder(
        self,
        *,
        folder: str,
        title: str,
    ) -> list[ManualImportCandidate] | None: ...

    @abstractmethod
    def history_for_download(self, *, download_id: str) -> HistoryPage | None: ...

    @abstractmethod
    def remote_path_mappings(self) -> list[RemotePathMapping] | None: ...

    @abstractmethod
    def manual_import_execute(
        self,
        *,
        files: list[ManualImportFile],
        import_mode: str = "auto",
    ) -> int | None: ...

    @abstractmethod
    def refresh_monitored_downloads(self) -> int | None: ...

    @abstractmethod
    def queue(self) -> list[QueueRecord] | None: ...

    @abstractmethod
    def queue_delete(self, queue_id: int) -> DeleteOutcome: ...

    @abstractmethod
    def quality_definitions(self) -> list[QualityDefinition]: ...

    @abstractmethod
    def languages(self) -> list[Language]: ...

    @abstractmethod
    def command_status(self, command_id: int) -> CommandResource: ...

    @abstractmethod
    def list_commands(self) -> list[CommandResource]: ...

    @abstractmethod
    def download_client_config(self) -> DownloadClientConfig: ...

    @abstractmethod
    def history_since(self, date: str) -> list[HistoryRecord] | None: ...


class SonarrClient(AbstractSonarrClient):
    """Thin client over the raw Sonarr v3 REST endpoints."""

    def __init__(
        self,
        *,
        http: ArrHttp,
        logger: logging.Logger,
    ) -> None:
        """Instantiate the Sonarr API client, network-free: the first request happens on the first method call."""

        self._http = http
        # No-retry clone for the wait-path polls: the import monitor loop IS the retry mechanism, so in-call
        # backoff only stretches each poll and multiplies identical warnings. `replace` shares the streak ledger.
        self._poll_http = replace(http, retries=0)
        self._logger = logger

    @override
    def all_series(self) -> list[SonarrItem]:
        """Every series in Sonarr (`/api/v3/series`, unfiltered).

        Fail-CLOSED: a failure raises rather than degrading to an empty library.
        """

        raw = self._http.get_json_list_strict("/api/v3/series")
        # Strict to match: a non-empty payload with zero valid records raises instead of reading as empty.
        return list[SonarrItem](validate_each(SonarrSeries, raw, strict=True))

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        """All episodes for a series, season/episode-sorted (`/api/v3/episode`), or None when unreachable."""

        warn = f"Could not fetch episodes for series {series_id} from Sonarr ({{detail}}) - skipping"
        raw = self._http.get_json_list(
            "/api/v3/episode",
            params={"seriesId": str(series_id), "includeImages": "false", "includeEpisodeFile": "true"},
            warn=None if quiet else warn,
        )
        if raw is None:
            return None

        episodes = validate_each(SonarrEpisode, raw)
        # A record missing either number sorts first (-1), never a None<int TypeError.
        episodes.sort(
            key=lambda ep: (
                ep.season_number if ep.season_number is not None else -1,
                ep.episode_number if ep.episode_number is not None else -1,
            ),
        )
        return episodes

    @override
    def parse(self, filename: str) -> SonarrParse | None:
        """Ask Sonarr to parse a filename (a basename, not a full path) into season/episode numbers.

        An empty episode list is a confirmed no-match (200, cacheable). None is a failure that must NOT be cached.
        """

        payload = self._http.get_json_dict(
            "/api/v3/parse",
            params={"title": filename},
            warn=f"Could not parse {filename} via Sonarr ({{detail}}) - skipping file",
        )
        if payload is None:
            return None

        # A present-but-non-list "episodes" is mangled, not a no-match: fail open to the uncacheable None.
        raw_eps = payload.get("episodes", [])
        if not isinstance(raw_eps, list):
            return None

        parsed: list[ParsedEpisode] = []
        for ep in cast("list[object]", raw_eps):
            try:
                record = _ParsedEpisode.model_validate(ep)
            except ValidationError:
                self._logger.debug(f"Sonarr's parse returned a malformed episode entry for {filename}; skipping it")
                continue

            if record.seasonNumber is None or record.episodeNumber is None:
                self._logger.debug(f"Sonarr's parse returned no season/episode number for {filename}; skipping it")
                continue

            parsed.append(ParsedEpisode(season=record.seasonNumber, episode=record.episodeNumber))

        # Coerced truthy to match ParsedFileInfo's BeforeValidator(bool). Absent or malformed reads False.
        info = payload.get("parsedEpisodeInfo")
        full_season = bool(info.get("fullSeason")) if is_json_obj(info) else False
        return SonarrParse(episodes=parsed, full_season=full_season)

    @override
    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None:
        """Parse a filename into season / episode / absolute numbers via `/api/v3/parse`.

        `parsedEpisodeInfo` carries series-agnostic numbers lifted from the release name, populated even when
        Sonarr matches no library series.
        """

        # Borrows the generous manual-import bound: this runs in the import wait over the same slow mount,
        # unlike the sweep's plain parse(). Rides the no-retry poll handle, the wait loop re-asks.
        payload = self._poll_http.get_json_dict(
            "/api/v3/parse",
            params={"title": filename},
            warn=f"Could not parse {filename} via Sonarr ({{detail}}) - will retry",
            timeout=MANUAL_IMPORT_TIMEOUT_S,
        )
        if payload is None:
            return None

        try:
            return ParsedFileInfo.model_validate(payload)
        except ValidationError as e:
            hub_warn(
                f"Could not parse {filename} via Sonarr (malformed response: {validation_summary(e)}) - will retry"
            )
            return None

    @override
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None:
        """List Sonarr's manual-import candidates for a completed download folder.

        Never send `seriesId`: Sonarr then scans the library folder rather than the download.
        """

        raw = self._poll_http.get_json_list(
            "/api/v3/manualimport",
            params={
                # Uppercased to match Sonarr's stored infohash form.
                "downloadId": pending.infohash.upper(),
                # Never filter existing files: a filtered scan drops the candidate for a file we mean to replace.
                "filterExistingFiles": "false",
            },
            warn=None,
            timeout=MANUAL_IMPORT_TIMEOUT_S,
        )
        if raw is None:
            return None
        return validate_each(ManualImportCandidate, raw)

    @override
    def manual_import_candidates_by_folder(
        self,
        *,
        folder: str,
        title: str,
    ) -> list[ManualImportCandidate] | None:
        """List manual-import candidates by scanning `folder` directly, which may be a single file path.

        Never send `downloadId` (the tracked scan 500s forever once history marks the download
        Imported/Failed/Ignored) or `seriesId` (it routes to a library-folder scan instead).
        """

        raw = self._poll_http.get_json_list(
            "/api/v3/manualimport",
            params={
                "folder": folder,
                # Same stance as the downloadId scan: never drop a candidate that would replace an existing file.
                "filterExistingFiles": "false",
            },
            warn=f"Could not fetch folder-scan import candidates for {title} ({{detail}}) - will retry",
            timeout=MANUAL_IMPORT_TIMEOUT_S,
        )
        if raw is None:
            return None
        return validate_each(ManualImportCandidate, raw)

    @override
    def history_for_download(self, *, download_id: str) -> HistoryPage | None:
        """Sonarr's history for one download, newest first (`/api/v3/history`).

        A paged envelope, unlike `/history/since`. None means no verdict, never clean history.
        """

        payload = self._poll_http.get_json_dict(
            "/api/v3/history",
            params={
                # Uppercased to match Sonarr's stored infohash form.
                "downloadId": download_id.upper(),
                "page": "1",
                "pageSize": "100",
                "sortKey": "date",
                "sortDirection": "descending",
            },
            warn="Could not read Sonarr's history for a download ({detail}) - assuming it is healthy",
        )
        if payload is None:
            return None

        try:
            return HistoryPage.model_validate(payload)
        except ValidationError as e:
            hub_warn(
                f"Could not read Sonarr's history for a download (malformed response: "
                f"{validation_summary(e)}) - assuming it is healthy"
            )
            return None

    @override
    def remote_path_mappings(self) -> list[RemotePathMapping] | None:
        """All Sonarr remote path mappings (`/api/v3/remotepathmapping`), or None on failure."""

        raw = self._http.get_json_list(
            "/api/v3/remotepathmapping",
            warn="Could not fetch Sonarr's remote path mappings ({detail}) - using download paths as-is",
        )
        if raw is None:
            return None
        return validate_each(RemotePathMapping, raw)

    @override
    def manual_import_execute(
        self,
        *,
        files: list[ManualImportFile],
        import_mode: str = "auto",
    ) -> int | None:
        """Queue a `ManualImport` command for the given files and return its command id.

        `import_mode` is Sonarr's `importMode`: `auto` (honors the copy/hardlink setting), `move`, or `copy`.
        """

        return self._post_command(CommandBody(name="ManualImport", importMode=import_mode, files=files))

    @override
    def refresh_monitored_downloads(self) -> int | None:
        """Queue Sonarr's `RefreshMonitoredDownloads` command and return its command id."""

        return self._post_command(CommandBody(name="RefreshMonitoredDownloads"))

    def _post_command(self, body: CommandBody) -> int | None:
        """POST a command to `/api/v3/command` and return its queued id, or None on failure.

        Never retried: a retry could double-queue the command.
        """

        # exclude_unset keeps exactly what the builder set (RefreshMonitoredDownloads stays a bare {"name"}),
        # never exclude_none: an explicitly set None must reach the wire.
        payload = self._http.post_json(
            "/api/v3/command",
            json=body.model_dump(exclude_unset=True),
            warn=f"Could not queue {body.name} command ({{detail}}) - will retry",
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            # A 2xx whose body carries no readable id: Sonarr may still have queued the command, so leave
            # a breadcrumb before reporting None.
            hub_warn(f"Could not confirm the {body.name} command was queued (unexpected payload) - will retry")
            return None

        # "id" defaults to 0 when absent, so it drops to None.
        try:
            command = CommandResource.model_validate(payload)
        except ValidationError as e:
            hub_warn(
                f"Could not confirm the {body.name} command was queued (malformed response: "
                f"{validation_summary(e)}) - will retry"
            )
            return None
        return command.id or None

    @override
    def queue(self) -> list[QueueRecord] | None:
        """All Sonarr queue records (`/api/v3/queue`, paged until `totalRecords` is covered), or None on failure.

        A season pack holds one record per episode sharing the `downloadId` (an infohash, case-insensitive).
        `includeUnknownSeriesItems` is on so an `importBlocked` item whose title matched no series still surfaces.
        """
        records: list[QueueRecord] = []
        page = 1
        while True:
            paged = self._http.get_json_dict(
                "/api/v3/queue",
                params={
                    "page": str(page),
                    "pageSize": "1000",
                    "includeUnknownSeriesItems": "true",
                },
                warn="Could not fetch the Sonarr queue ({detail}) - will retry",
            )
            if paged is None:
                # ANY failed page fails the whole read: a partial queue misreads a tracked download
                # as untracked, so the caller must retry rather than step in.
                return None

            raw = paged.get("records")
            page_records = validate_each(QueueRecord, cast("list[object]", raw)) if isinstance(raw, list) else []
            records.extend(page_records)

            total = paged.get("totalRecords")
            if not page_records or len(records) >= (total if isinstance(total, int) else 0):
                return records
            page += 1

    @override
    def queue_delete(self, queue_id: int) -> DeleteOutcome:
        """Dismiss one queue item (`DELETE /api/v3/queue/{id}`), and with it the whole tracked download.

        `removeFromClient=false` + `blocklist=false`: Sonarr durably marks the download manually ignored (never
        importing it again) while the torrent keeps seeding in qBittorrent, unblocklisted and not re-searched.
        """

        return self._http.delete(
            f"/api/v3/queue/{queue_id}",
            params={"removeFromClient": "false", "blocklist": "false"},
            warn="Could not remove the finished download from Sonarr's queue ({detail})",
        )

    @override
    def quality_definitions(self) -> list[QualityDefinition]:
        """All Sonarr quality definitions (`/api/v3/qualitydefinition`), or an empty list on failure."""

        raw = self._http.get_json_list(
            "/api/v3/qualitydefinition",
            warn="Could not fetch quality definitions from Sonarr ({detail}) - falling back to other quality sources",
        )
        if raw is None:
            return []
        # The nested quality keeps its unknown keys, so the manual-import payload can re-emit it whole.
        return validate_each(QualityDefinition, raw)

    @override
    def languages(self) -> list[Language]:
        """All Sonarr languages (`/api/v3/language`), or an empty list on failure."""

        raw = self._http.get_json_list(
            "/api/v3/language",
            warn="Could not fetch languages from Sonarr ({detail}) - using the candidate's languages",
        )
        if raw is None:
            return []
        return validate_each(Language, raw)

    @override
    def command_status(self, command_id: int) -> CommandResource:
        """Current state of a Sonarr command (`/api/v3/command/{id}`), or a blank `CommandResource` on failure."""

        payload = self._http.get_json_dict(
            f"/api/v3/command/{command_id}",
            warn=f"Could not fetch status for command {command_id} ({{detail}}) - leaving the import unverified",
        )
        if payload is None:
            return CommandResource()

        # A malformed body fails open to the same default (status None) as a transport miss. The refresh
        # poll loop depends on it.
        try:
            return CommandResource.model_validate(payload)
        except ValidationError as e:
            hub_warn(
                f"Could not read status for command {command_id} ({validation_summary(e)}) - "
                "leaving the import unverified"
            )
            return CommandResource()

    @override
    def list_commands(self) -> list[CommandResource]:
        """All Sonarr commands (`/api/v3/command`), or an empty list on failure."""

        raw = self._http.get_json_list(
            "/api/v3/command",
            warn="Could not fetch the Sonarr command list ({detail}) - assuming nothing is in flight",
        )
        if raw is None:
            return []
        return validate_each(CommandResource, raw)

    @override
    def download_client_config(self) -> DownloadClientConfig:
        """Sonarr's download client config (`/api/v3/config/downloadclient`), or the defaults on failure."""

        warn = "Could not read Sonarr's download client config ({detail}) - assuming completed download handling is on"
        payload = self._http.get_json_dict("/api/v3/config/downloadclient", warn=warn)
        if payload is None:
            return DownloadClientConfig()
        try:
            return DownloadClientConfig.model_validate(payload)
        except ValidationError as e:
            hub_warn(warn.format(detail=f"malformed response: {validation_summary(e)}"))
            return DownloadClientConfig()

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History since `date`, or None on failure."""

        return self._http.history_since(
            date,
            include_flags={"includeSeries": "false", "includeEpisode": "false"},
        )
