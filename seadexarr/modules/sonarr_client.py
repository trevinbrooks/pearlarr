"""Sonarr REST client: the HTTP surface the Sonarr syncer talks to.

``SonarrClient`` speaks to the raw ``/api/v3`` endpoints directly, every one
riding the httpx-based :class:`~.arr_http.ArrHttp` bound at construction.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast, override

from pydantic import BaseModel, ConfigDict, ValidationError

from .arr_http import ArrHttp
from .log import indent_string
from .manual_import import PendingImport
from .seadex_types import (
    CommandBody,
    CommandResource,
    HistoryRecord,
    Json,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
    validate_each,
    validation_summary,
)

# Per-request timeout (seconds) for the manual-import folder scan. Sonarr walks
# and parses every file under the folder, which is slow - and can hang - over a
# remote mount; bounding it lets a hung scan surface as a transient miss (retry)
# instead of blocking the whole run. Generous so a legitimately slow first scan
# (uncached remote files) still completes.
MANUAL_IMPORT_TIMEOUT_S = 120


class _ParsedEpisode(BaseModel):
    """One ``ParseResource.episodes[]`` entry, reduced to the two numbers read.

    Private to :meth:`SonarrClient.parse` - the only consumer of the
    series-matched array (the file size comes from the SeaDex file list).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    seasonNumber: int | None = None
    episodeNumber: int | None = None


class AbstractSonarrClient(ABC):
    """The Sonarr read/command surface the four Sonarr collaborators consume.

    A nominal seam (``cosmicpython``'s ``AbstractRepository`` pattern) over the
    public methods the episode / parse / mapper / import collaborators call
    on their injected ``sonarr``. Both the real :class:`SonarrClient` and the test
    ``FakeSonarrClient`` subclass it, so an incomplete fake is a static
    ``reportAbstractUsage`` error *and* an un-instantiable ``TypeError`` - the
    collaborators take this type, never the concrete client, so a fake is checked
    against the real surface at the injection seam.
    """

    @abstractmethod
    def all_series(self) -> list[SonarrItem]: ...

    @abstractmethod
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None: ...

    @abstractmethod
    def parse(self, filename: str) -> list[dict[str, int]] | None: ...

    @abstractmethod
    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None: ...

    @abstractmethod
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None: ...

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
    def queue(self) -> list[QueueRecord]: ...

    @abstractmethod
    def quality_definitions(self) -> list[QualityDefinition]: ...

    @abstractmethod
    def languages(self) -> list[Language]: ...

    @abstractmethod
    def command_status(self, command_id: int) -> CommandResource: ...

    @abstractmethod
    def list_commands(self) -> list[CommandResource]: ...

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
        """Instantiate the Sonarr API client.

        Construction is network-free (no connection probe): the first request
        happens on the first method call, so an unreachable Sonarr surfaces as
        that call's typed error / fail-open path, never a constructor hang.

        Args:
            http (ArrHttp): The transport already bound to Sonarr's url + key
                (``ArrHttp.bind`` with ``label="Sonarr"``).
            logger (logging.Logger): For the client's own non-transport warnings
                (the bound transport carries its own logger for request lines).
        """

        self._http = http
        self._logger = logger

    @override
    def all_series(self) -> list[SonarrItem]:
        """Every series in Sonarr (``/api/v3/series``, unfiltered).

        The one fail-CLOSED read: the library list is the run's ground truth
        (an outage reading as an empty library would silently no-op the leg),
        so a failure raises the typed :mod:`~.arr_http` errors for the CLI
        containment arms instead of degrading to an empty list.
        """

        raw = self._http.get_json_list_strict("/api/v3/series")
        # Strict validation to match: a non-empty payload with zero valid
        # records raises BoundaryContractError instead of reading as empty.
        return list[SonarrItem](validate_each(SonarrSeries, raw, logger=self._logger, strict=True))

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        """All episodes for a series, season/episode-sorted (``/api/v3/episode``).

        Returns None (with a warning) if Sonarr is unreachable, so the caller can
        skip the id gracefully. Stateless over the shared httpx client, so the
        concurrent prefetch can call it from worker threads.

        Args:
            series_id (int): Series ID in Sonarr.
            quiet (bool): Suppress the unreachable-warning. The concurrent
                episode prefetch passes this so a transient miss isn't logged
                from a worker thread - it is retried, and logged if it still
                fails, on the main thread when ``get_ep_list`` re-fetches.
        """

        warn = f"Could not fetch episodes for series {series_id} from Sonarr ({{detail}}); skipping"
        raw = self._http.get_json_list(
            "/api/v3/episode",
            params={"seriesId": str(series_id), "includeImages": "false", "includeEpisodeFile": "true"},
            warn=None if quiet else warn,
        )
        if raw is None:
            return None

        # Validate each record at this client boundary (junk records skip with a
        # warning), then sort by season/episode for slicing later. A record
        # missing either number sorts first (-1), never a None<int TypeError.
        episodes = validate_each(SonarrEpisode, raw, logger=self._logger)
        episodes.sort(
            key=lambda ep: (
                ep.season_number if ep.season_number is not None else -1,
                ep.episode_number if ep.episode_number is not None else -1,
            ),
        )
        return episodes

    @override
    def parse(self, filename: str) -> list[dict[str, int]] | None:
        """Ask Sonarr to parse a single filename into season/episode numbers.

        Only the season/episode mapping is returned - the file size is filled in
        by the caller, since it comes from the SeaDex file list rather than from
        Sonarr.

        Distinguishes a clean response from a transient failure so the caller can
        safely negative-cache the former without poisoning the latter: an empty
        list is a *confirmed* "Sonarr matched no episode" (200, cacheable),
        whereas None is a request failure (non-200 / connection error after
        ArrHttp's retries) that must NOT be cached.

        Args:
            filename (str): Filename to parse (basename, not full path).

        Returns:
            list[dict[str, int]] | None: {"season", "episode"} dicts on a clean
                200 (empty when Sonarr genuinely matched nothing), or None when
                the request failed.
        """

        payload = self._http.get_json_dict(
            "/api/v3/parse",
            params={"title": filename},
            warn=indent_string(f"Could not parse {filename} via Sonarr ({{detail}}); skipping file"),
        )
        if payload is None:
            return None

        # The ParseResource's "episodes" is an array of EpisodeResource objects,
        # validated per entry at this boundary. A present-but-non-list
        # "episodes" is a mangled response, not a confirmed no-match: fail open
        # to the uncacheable None.
        raw_eps = payload.get("episodes", [])
        if not isinstance(raw_eps, list):
            return None

        parsed: list[dict[str, int]] = []
        for ep in cast("list[object]", raw_eps):
            try:
                record = _ParsedEpisode.model_validate(ep)
            except ValidationError:
                # A junk-typed entry skips like a missing-number one below.
                self._logger.debug(
                    indent_string(f"Sonarr's parse returned a malformed episode entry for {filename}; skipping it"),
                )
                continue

            if record.seasonNumber is None or record.episodeNumber is None:
                self._logger.debug(
                    indent_string(
                        f"Sonarr's parse returned no season/episode number for {filename}; skipping it",
                    ),
                )
                continue

            parsed.append({"season": record.seasonNumber, "episode": record.episodeNumber})

        return parsed

    @override
    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None:
        """Parse a filename into SERIES-AGNOSTIC season / episode / absolute numbers.

        Reads the ``/api/v3/parse`` response's ``parsedEpisodeInfo`` - the numbers
        Sonarr lifts straight from the release NAME - rather than its ``episodes``
        array (Sonarr's series-*matched* episodes, which is empty whenever the
        title can't be matched to a library series). That field is what lets the
        import place a specials / alias-titled release Sonarr can't match: the
        season+episode (or absolute) numbers are still present, and OUR resolved
        mapping turns them into episode ids.

        Returns the parsed info, or None (with a warning) on a non-200 or a
        transient request error, so the caller can retry.

        Args:
            filename (str): Filename to parse (basename, not full path).
        """

        payload = self._http.get_json_dict(
            "/api/v3/parse",
            params={"title": filename},
            warn=indent_string(f"Could not parse {filename} via Sonarr ({{detail}}); will retry"),
            timeout=MANUAL_IMPORT_TIMEOUT_S,
        )
        if payload is None:
            return None

        # The ParseResource's parsedEpisodeInfo carries the series-agnostic
        # numbers; a malformed body fails open to the same retryable None.
        try:
            return ParsedFileInfo.model_validate(payload)
        except ValidationError as e:
            self._logger.warning(
                indent_string(f"Could not parse {filename} via Sonarr (malformed response: {validation_summary(e)})"),
            )
            return None

    @override
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None:
        """List Sonarr's manual-import candidates for a completed download folder.

        Scans by ``downloadId`` only (no ``seriesId``): we consume the candidates'
        on-disk ``path`` + ``quality`` and assign episode identity ourselves from
        OUR resolved mapping, so Sonarr's own title parse is irrelevant here - a
        candidate Sonarr rejects as "Unknown Series" still gives us its path, which
        is all this call is for. (``seriesId`` is deliberately NOT sent: pinning it
        makes Sonarr scan the *library* folder rather than the download, returning
        the wrong files.)

        Returns ``None`` (with a warning) on a non-200 *or* a transient request
        error (timeout / connection drop) - both mean "ask again", e.g. Sonarr is
        still building the parse over a slow remote mount. Returns an empty list
        only when Sonarr genuinely reports no candidates (the files aren't visible
        on its mount yet). The caller treats both as keep-waiting, but the
        distinction keeps the intent clear.

        Each raw ManualImportResource is narrowed to a
        :class:`~.seadex_types.ManualImportCandidate` (``path`` / ``quality`` /
        ``rejections``) via its ``from_api`` at this client boundary, so the
        decision path never touches the raw DTO.

        Args:
            pending (PendingImport): The pending import record to scan for

        Returns:
            list[ManualImportCandidate] | None: The parsed candidates; ``None`` on
                a transient failure.
        """

        raw = self._http.get_json_list(
            "/api/v3/manualimport",
            params={
                "downloadId": pending.infohash.upper(),
                # Never filter existing files: our import may replace an episode's
                # non-recommended file, whose candidate a filtered scan would drop.
                "filterExistingFiles": "false",
            },
            warn=indent_string(
                f"Could not fetch manual-import candidates for {pending.title} ({{detail}}); will retry"
            ),
            timeout=MANUAL_IMPORT_TIMEOUT_S,
        )
        if raw is None:
            return None

        # Each element is an unvalidated ManualImportResource: cast at the parse
        # boundary (skipping strays), then narrow to the fields planning reads.
        return [ManualImportCandidate.from_api(cast("dict[str, Any]", c)) for c in raw if isinstance(c, dict)]

    @override
    def manual_import_execute(
        self,
        *,
        files: list[ManualImportFile],
        import_mode: str = "auto",
    ) -> int | None:
        """Queue a ``ManualImport`` command for the given files (no title parse).

        Each entry in ``files`` carries the authoritative mapping we computed
        (``seriesId``, ``episodeIds``, ``releaseGroup``, ``quality`` ...), so
        Sonarr imports without re-deriving anything from the release title.

        Returns the command ``id`` (for optional completion verification) or
        None (with a warning) on failure, so the caller can leave the import
        pending and retry later.

        Args:
            files (list[ManualImportFile]): ManualImport file payloads to import.
            import_mode (str): Sonarr ``importMode``: ``auto`` (default; respects
                the copy/hardlink setting and preserves seeding), ``move`` or
                ``copy``.

        Returns:
            int | None: The queued command's id, or None on failure.
        """

        body: CommandBody = {
            "name": "ManualImport",
            "importMode": import_mode,
            "files": files,
        }
        return self._post_command(body)

    @override
    def refresh_monitored_downloads(self) -> int | None:
        """Queue Sonarr's ``RefreshMonitoredDownloads`` command.

        Makes Sonarr re-scan its download clients (picking up our completed
        torrent and refreshing the remote mount path) and re-evaluate its queue,
        so the queue's ``trackedDownloadState`` reflects reality before we read it.

        Returns the command ``id`` (poll :meth:`command_status` to wait for it) or
        None on failure.
        """

        body: CommandBody = {"name": "RefreshMonitoredDownloads"}
        return self._post_command(body)

    def _post_command(self, body: CommandBody) -> int | None:
        """POST a command to ``/api/v3/command`` and return its queued id.

        Shared by :meth:`manual_import_execute` and
        :meth:`refresh_monitored_downloads`. Returns the command ``id`` or None
        (with a warning) on a non-2xx; the POST rides :meth:`~.arr_http.ArrHttp.post_json`,
        so it is never retried (a retry could double-queue the command).

        Args:
            body (CommandBody): The outgoing command body (must carry ``name``).
        """

        # CommandBody is JSON-safe but a non-closed TypedDict can never be
        # proven against the recursive alias, so cast at the wire boundary.
        payload = self._http.post_json(
            "/api/v3/command",
            json=cast("Json", body),
            warn=indent_string(f"Could not queue {body['name']} command ({{detail}})"),
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            # A 2xx whose body carries no readable id: Sonarr may still have
            # queued the command, so leave a breadcrumb before reporting None.
            self._logger.warning(indent_string(f"Could not queue {body['name']} command (unexpected payload)"))
            return None

        # The returned CommandResource's "id" is the queued command id (0 when
        # absent, so the caller drops it); a malformed body fails open to None.
        try:
            command = CommandResource.model_validate(payload)
        except ValidationError as e:
            self._logger.warning(
                indent_string(f"Could not queue {body['name']} command (malformed response: {validation_summary(e)})"),
            )
            return None
        return command.id or None

    @override
    def queue(self) -> list[QueueRecord]:
        """All Sonarr queue records (``/api/v3/queue``).

        Used to see what Sonarr is doing with a download we added directly to
        qBittorrent: each record carries ``downloadId`` (the infohash, matched
        case-insensitively) and ``trackedDownloadState``. A season pack has one
        record per episode sharing the ``downloadId``. ``includeUnknownSeriesItems``
        is on because an ``importBlocked`` item whose title didn't match a series
        can surface as an unknown-series record. Pages of 1000 are fetched until
        ``totalRecords`` is covered, so a very large queue is never silently
        truncated.

        Each raw ``QueueResource`` is narrowed to a
        :class:`~.seadex_types.QueueRecord` (``download_id`` / ``state`` /
        ``status``) via its ``from_api`` at this client boundary, so the wait
        decision never touches the raw DTO.

        Returns an empty list (with a warning) on a non-200, so the caller treats
        "couldn't read the queue" as "not tracked" and falls back to its own scan.

        Returns:
            list[QueueRecord]: The parsed queue records; empty on failure.
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
                warn=indent_string("Could not fetch the Sonarr queue ({detail})"),
            )
            if paged is None:
                # A failed LATER page keeps what was fetched: partial beats empty
                # for the caller's "not tracked -> fall back to own scan" logic.
                return records

            # The paged object's "records" is the array of QueueResource objects;
            # validate each at this boundary (a stray non-object entry is skipped
            # with a warning, never crashed on).
            raw = paged.get("records")
            page_records = (
                validate_each(QueueRecord, cast("list[object]", raw), logger=self._logger)
                if isinstance(raw, list)
                else []
            )
            records.extend(page_records)

            total = paged.get("totalRecords")
            if not page_records or len(records) >= (total if isinstance(total, int) else 0):
                return records
            page += 1

    @override
    def quality_definitions(self) -> list[QualityDefinition]:
        """All Sonarr quality definitions (``/api/v3/qualitydefinition``).

        Used to resolve a quality NAME (e.g. ``Bluray-2160p``) to a Sonarr
        QualityModel for the manual-import payload.

        Returns an empty list (with a warning) on a non-200, so the caller can
        fall back to other quality sources.

        Returns:
            list[QualityDefinition]: Raw QualityDefinitionResource dicts (each
                wraps a nested ``quality`` object the resolver re-emits verbatim);
                empty on failure.
        """

        raw = self._http.get_json_list(
            "/api/v3/qualitydefinition",
            warn="Could not fetch quality definitions from Sonarr ({detail})",
        )
        if raw is None:
            return []

        # QualityDefinitionResource dicts pass through verbatim (the resolver
        # re-emits the nested "quality"): cast at the parse boundary, skip strays.
        return [cast("QualityDefinition", record) for record in raw if isinstance(record, dict)]

    @override
    def languages(self) -> list[Language]:
        """All Sonarr languages (``/api/v3/language``).

        Used to resolve language names to ``{id, name}`` objects for the
        manual-import payload.

        Returns an empty list (with a warning) on a non-200, so the caller can
        fall back to the candidate's languages.

        Returns:
            list[Language]: Raw LanguageResource dicts (the ``{id, name}`` the
                resolver matches by name and re-emits verbatim); empty on failure.
        """

        raw = self._http.get_json_list(
            "/api/v3/language",
            warn="Could not fetch languages from Sonarr ({detail})",
        )
        if raw is None:
            return []

        # LanguageResource dicts ({id, name}) pass through verbatim (the resolver
        # matches by name and re-emits): cast at the parse boundary, skip strays.
        return [cast("Language", record) for record in raw if isinstance(record, dict)]

    @override
    def command_status(self, command_id: int) -> CommandResource:
        """Current state of a Sonarr command (``/api/v3/command/{id}``).

        Used by :meth:`~.sonarr_import.ImportExecutor.refresh_downloads` to poll a
        queued ``RefreshMonitoredDownloads`` command until it finishes.

        Returns a default :class:`~.seadex_types.CommandResource` (with a warning)
        on a non-200, so the caller can treat the import as unverified and leave it
        pending.

        Args:
            command_id (int): Command ID returned by ``manual_import_execute``.

        Returns:
            CommandResource: The command's parsed state (``status`` / ``result``);
                a default (``status`` None) on failure.
        """

        payload = self._http.get_json_dict(
            f"/api/v3/command/{command_id}",
            warn=indent_string(f"Could not fetch status for command {command_id} ({{detail}})"),
        )
        if payload is None:
            return CommandResource()

        # A malformed body fails open to the same default (status None) the
        # transport miss takes - the refresh poll loop depends on it.
        try:
            return CommandResource.model_validate(payload)
        except ValidationError as e:
            self._logger.warning(
                indent_string(f"Could not read status for command {command_id} ({validation_summary(e)})"),
            )
            return CommandResource()

    @override
    def list_commands(self) -> list[CommandResource]:
        """All Sonarr commands (``/api/v3/command``).

        Used by the in-flight ManualImport guard to see whether a ManualImport we
        (or a prior run) POSTed for a download is still ``queued``/``started`` -
        so we don't stack a duplicate while Sonarr is already importing it. Each
        raw ``CommandResource`` is narrowed via its
        :meth:`~.seadex_types.CommandResource.from_api` (``name`` / ``status`` /
        ``message`` / ``body.files``) at this client boundary, mirroring
        :meth:`queue`.

        Returns an empty list (with a warning) on a non-200, so the caller treats
        "couldn't read the commands" as "nothing in flight" and proceeds (a false
        step-in is bounded by the import deadline, a missed read just re-checks
        next poll).

        Returns:
            list[CommandResource]: The parsed commands; empty on failure.
        """

        raw = self._http.get_json_list(
            "/api/v3/command",
            warn=indent_string("Could not fetch the Sonarr command list ({detail})"),
        )
        if raw is None:
            return []

        # Validate each command at this boundary (strays skip with a warning).
        return validate_each(CommandResource, raw, logger=self._logger)

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History since ``date``, or None on failure (fail-open; shared helper)."""

        return self._http.history_since(
            date,
            include_flags={"includeSeries": "false", "includeEpisode": "false"},
        )
