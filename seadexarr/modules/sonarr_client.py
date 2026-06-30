"""Sonarr REST client: the HTTP surface the Sonarr syncer talks to.

``SonarrClient`` wraps the high-level ``arrapi`` client (``all_series``)
and the two raw endpoints the syncer needs
(``/api/v3/episode`` and ``/api/v3/parse``)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast, override
from urllib.parse import urlencode

import requests
from arrapi import SonarrAPI

from .log import indent_string
from .manual_import import PendingImport
from .seadex_types import (
    CommandBody,
    CommandResource,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    ParsedEpisode,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    SonarrEpisode,
    SonarrItem,
)

# Per-request timeout (seconds) for the manual-import folder scan. Sonarr walks
# and parses every file under the folder, which is slow - and can hang - over a
# remote mount; bounding it lets a hung scan surface as a transient miss (retry)
# instead of blocking the whole run. Generous so a legitimately slow first scan
# (uncached remote files) still completes.
MANUAL_IMPORT_TIMEOUT_S = 120

# (connect, read) timeout for the episode/parse GETs, so a hung Sonarr surfaces
# as a transient miss rather than blocking the (now concurrent) sweep.
SONARR_REQUEST_TIMEOUT_S = (5, 30)


class AbstractSonarrClient(ABC):
    """The Sonarr read/command surface the four Sonarr collaborators consume.

    A nominal seam (``cosmicpython``'s ``AbstractRepository`` pattern) over the
    twelve public methods the episode / parse / mapper / import collaborators call
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
        filter_existing_files: bool = False,
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


class SonarrClient(AbstractSonarrClient):
    """Thin wrapper over the Sonarr API (``arrapi`` + two raw endpoints)."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        session: requests.Session,
        logger: logging.Logger,
    ) -> None:
        """Instantiate the Sonarr API client.

        Args:
            url (str): Sonarr base URL.
            api_key (str): Sonarr API key.
            session (requests.Session): Shared keep-alive session for the raw
                endpoints. ``parse`` fires one request per file, so reusing it
                removes a per-file handshake.
            logger (logging.Logger): For request warnings.
        """

        self._url = url
        self._api_key = api_key
        self._session = session
        self._logger = logger
        self._api = SonarrAPI(url=url, apikey=api_key)

    @override
    def all_series(self) -> list[SonarrItem]:
        """Every series in Sonarr (unfiltered)."""

        # arrapi ships no py.typed, so all_series() is Unknown; the series
        # objects expose the attribute surface of SonarrItem, so cast at this
        # client boundary into the project's typed shape.
        return cast("list[SonarrItem]", self._api.all_series())

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        """All episodes for a series, season/episode-sorted (``/api/v3/episode``).

        Returns None (with a warning) if Sonarr is unreachable, so the caller can
        skip the id gracefully.

        Args:
            series_id (int): Series ID in Sonarr.
            quiet (bool): Suppress the unreachable-warning. The concurrent
                episode prefetch passes this so a transient miss isn't logged
                from a worker thread - it is retried, and logged if it still
                fails, on the main thread when ``get_ep_list`` re-fetches.
        """

        eps_req_url = (
            f"{self._url}/api/v3/episode?"
            f"seriesId={series_id}&"
            f"includeImages=false&"
            f"includeEpisodeFile=true&"
            f"apikey={self._api_key}"
        )
        try:
            eps_req = self._session.get(eps_req_url, timeout=SONARR_REQUEST_TIMEOUT_S)
        except requests.RequestException:
            eps_req = None

        if eps_req is None or eps_req.status_code != 200:
            if not quiet:
                self._logger.warning(
                    "Could not fetch episode data from Sonarr; it may be unreachable",
                )
            return None

        # response.json() is untyped; the episode endpoint returns a JSON array
        # of objects, so cast at the parse boundary before sorting/parsing.
        raw_json = cast("list[dict[str, Any]]", eps_req.json())

        # Sort by season/episode number for slicing later, then parse each raw
        # record into a SonarrEpisode at this client boundary.
        raw_eps = sorted(
            raw_json,
            key=lambda x: (
                x.get("seasonNumber", None),
                x.get("episodeNumber", None),
            ),
        )
        return [SonarrEpisode.from_api(ep) for ep in raw_eps]

    @override
    def parse(self, filename: str) -> list[dict[str, int]] | None:
        """Ask Sonarr to parse a single filename into season/episode numbers.

        Only the season/episode mapping is returned - the file size is filled in
        by the caller, since it comes from the SeaDex file list rather than from
        Sonarr.

        Distinguishes a clean response from a transient failure so the caller can
        safely negative-cache the former without poisoning the latter: an empty
        list is a *confirmed* "Sonarr matched no episode" (200, cacheable),
        whereas None is a request failure (non-200 / connection error after the
        session's retries) that must NOT be cached.

        Args:
            filename (str): Filename to parse (basename, not full path).

        Returns:
            list[dict[str, int]] | None: {"season", "episode"} dicts on a clean
                200 (empty when Sonarr genuinely matched nothing), or None when
                the request failed.
        """

        d = {"title": filename, "apikey": self._api_key}
        d_enc = urlencode(d)

        # Parse through Sonarr
        parse_req_url = f"{self._url}/api/v3/parse?{d_enc}"
        try:
            parse_req = self._session.get(parse_req_url, timeout=SONARR_REQUEST_TIMEOUT_S)
        except requests.RequestException:
            self._logger.warning(
                indent_string(
                    f"Could not parse {filename} via Sonarr (request failed); skipping file",
                ),
            )
            return None

        if parse_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not parse {filename} via Sonarr (status code {parse_req.status_code}); skipping file",
                ),
            )
            return None

        # response.json() is untyped; the parse endpoint returns a ParseResource
        # JSON object whose "episodes" is an array of EpisodeResource objects, so
        # cast at the parse boundary before reading their season/episode numbers.
        parse_body = cast("dict[str, list[ParsedEpisode]]", parse_req.json())
        episode_info: list[ParsedEpisode] = parse_body.get("episodes", [])

        parsed: list[dict[str, int]] = []
        for ep in episode_info:
            season = ep.get("seasonNumber", None)
            episode = ep.get("episodeNumber", None)

            if season is None or episode is None:
                self._logger.debug(
                    indent_string(
                        f"Season or episode came up None for {filename}; skipping this episode entry",
                    ),
                )
                continue

            parsed.append({"season": season, "episode": episode})

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

        d_enc = urlencode({"title": filename, "apikey": self._api_key})
        parse_req_url = f"{self._url}/api/v3/parse?{d_enc}"
        try:
            parse_req = self._session.get(parse_req_url, timeout=MANUAL_IMPORT_TIMEOUT_S)
        except requests.RequestException as e:
            self._logger.warning(
                indent_string(f"Could not parse {filename} via Sonarr ({e}); will retry"),
            )
            return None

        if parse_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not parse {filename} via Sonarr (status code {parse_req.status_code}); will retry",
                ),
            )
            return None

        # response.json() is untyped; the parse endpoint returns a ParseResource
        # whose parsedEpisodeInfo carries the series-agnostic numbers - narrow it
        # to ParsedFileInfo at this client boundary.
        parse_body = cast("dict[str, Any]", parse_req.json())
        return ParsedFileInfo.from_parse_resource(parse_body)

    @override
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
        filter_existing_files: bool = False,
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
            filter_existing_files (bool): If True, Sonarr drops files it already
                has imported. Sent lowercase ``true``/``false``.

        Returns:
            list[ManualImportCandidate] | None: The parsed candidates; ``None`` on
                a transient failure.
        """

        params: dict[str, str] = {
            "downloadId": pending.infohash.upper(),
            "filterExistingFiles": "true" if filter_existing_files else "false",
            "apikey": self._api_key,
        }
        params_enc = urlencode(params)

        candidates_req_url = f"{self._url}/api/v3/manualimport?{params_enc}"
        try:
            candidates_req = self._session.get(
                candidates_req_url,
                timeout=MANUAL_IMPORT_TIMEOUT_S,
            )
        except requests.RequestException as e:
            self._logger.warning(
                indent_string(
                    f"Manual-import scan of {pending.title} did not respond ({e}); will retry",
                ),
            )
            return None

        if candidates_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch manual-import candidates for "
                    f"{pending.title} (status code {candidates_req.status_code}); "
                    f"will retry",
                ),
            )
            return None

        # response.json() is untyped; the manualimport endpoint returns a JSON
        # array of ManualImportResource objects, so cast at the parse boundary,
        # then narrow each to the fields planning reads via from_api.
        raw_candidates = cast("list[dict[str, Any]]", candidates_req.json())
        return [ManualImportCandidate.from_api(c) for c in raw_candidates]

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
        return self._post_command(body, label="ManualImport")

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
        return self._post_command(body, label="RefreshMonitoredDownloads")

    def _post_command(self, body: CommandBody, *, label: str) -> int | None:
        """POST a command to ``/api/v3/command`` and return its queued id.

        Shared by :meth:`manual_import_execute` and
        :meth:`refresh_monitored_downloads`. Returns the command ``id`` or None
        (with a warning) on a non-2xx.

        Args:
            body (CommandBody): The outgoing command body (must carry ``name``).
            label (str): Command name for the warning message.
        """

        d_enc = urlencode({"apikey": self._api_key})
        command_req_url = f"{self._url}/api/v3/command?{d_enc}"
        command_req = self._session.post(command_req_url, json=body)

        if command_req.status_code not in (200, 201):
            self._logger.warning(
                indent_string(
                    f"Could not queue {label} command (status code {command_req.status_code})",
                ),
            )
            return None

        # response.json() is untyped; the command POST returns a CommandResource
        # JSON object whose "id" is the queued command id (0 when absent), so cast
        # at the parse boundary and narrow it to the consumed fields.
        command = CommandResource.from_api(cast("dict[str, Any]", command_req.json()))
        return command.id or None

    @override
    def queue(self) -> list[QueueRecord]:
        """All Sonarr queue records (``/api/v3/queue``).

        Used to see what Sonarr is doing with a download we added directly to
        qBittorrent: each record carries ``downloadId`` (the infohash, matched
        case-insensitively) and ``trackedDownloadState``. A season pack has one
        record per episode sharing the ``downloadId``. ``includeUnknownSeriesItems``
        is on because an ``importBlocked`` item whose title didn't match a series
        can surface as an unknown-series record. A large ``pageSize`` pulls the
        whole queue in one request.

        Each raw ``QueueResource`` is narrowed to a
        :class:`~.seadex_types.QueueRecord` (``download_id`` / ``state`` /
        ``status``) via its ``from_api`` at this client boundary, so the wait
        decision never touches the raw DTO.

        Returns an empty list (with a warning) on a non-200, so the caller treats
        "couldn't read the queue" as "not tracked" and falls back to its own scan.

        Returns:
            list[QueueRecord]: The parsed queue records; empty on failure.
        """
        params = urlencode(
            {
                "pageSize": "1000",
                "includeUnknownSeriesItems": "true",
                "apikey": self._api_key,
            },
        )
        queue_req_url = f"{self._url}/api/v3/queue?{params}"
        queue_req = self._session.get(queue_req_url)

        if queue_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch the Sonarr queue (status code {queue_req.status_code})",
                ),
            )
            return []

        # response.json() is untyped; the queue endpoint returns a paged object
        # whose "records" is the array of QueueResource objects, so cast at the
        # parse boundary, then narrow each to the fields the wait reads via
        # from_api.
        paged = cast("dict[str, list[dict[str, Any]]]", queue_req.json())
        return [QueueRecord.from_api(record) for record in paged.get("records", [])]

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

        defs_req_url = f"{self._url}/api/v3/qualitydefinition?apikey={self._api_key}"
        defs_req = self._session.get(defs_req_url)

        if defs_req.status_code != 200:
            self._logger.warning(
                "Could not fetch quality definitions from Sonarr; it may be unreachable",
            )
            return []

        # response.json() is untyped; the endpoint returns a JSON array of
        # QualityDefinitionResource objects whose nested "quality" the resolver
        # re-emits verbatim, so cast at the parse boundary.
        return cast("list[QualityDefinition]", defs_req.json())

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

        langs_req_url = f"{self._url}/api/v3/language?apikey={self._api_key}"
        langs_req = self._session.get(langs_req_url)

        if langs_req.status_code != 200:
            self._logger.warning(
                "Could not fetch languages from Sonarr; it may be unreachable",
            )
            return []

        # response.json() is untyped; the endpoint returns a JSON array of
        # LanguageResource objects ({id, name}) the resolver re-emits verbatim, so
        # cast at the parse boundary.
        return cast("list[Language]", langs_req.json())

    @override
    def command_status(self, command_id: int) -> CommandResource:
        """Current state of a Sonarr command (``/api/v3/command/{id}``).

        Used to optionally verify a ``ManualImport`` completed before the caller
        removes the pending record.

        Returns a default :class:`~.seadex_types.CommandResource` (with a warning)
        on a non-200, so the caller can treat the import as unverified and leave it
        pending.

        Args:
            command_id (int): Command ID returned by ``manual_import_execute``.

        Returns:
            CommandResource: The command's parsed state (``status`` / ``result``);
                a default (``status`` None) on failure.
        """

        status_req_url = f"{self._url}/api/v3/command/{command_id}?apikey={self._api_key}"
        status_req = self._session.get(status_req_url)

        if status_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch status for command {command_id} (status code {status_req.status_code})",
                ),
            )
            return CommandResource()

        # response.json() is untyped; the endpoint returns a single
        # CommandResource JSON object, so cast at the parse boundary and narrow it
        # to the consumed fields.
        return CommandResource.from_api(cast("dict[str, Any]", status_req.json()))

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

        commands_req_url = f"{self._url}/api/v3/command?apikey={self._api_key}"
        commands_req = self._session.get(commands_req_url)

        if commands_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch the Sonarr command list (status code {commands_req.status_code})",
                ),
            )
            return []

        # response.json() is untyped; the command endpoint returns a JSON array of
        # CommandResource objects, so cast at the parse boundary, then narrow each
        # to the fields the guard reads via from_api.
        raw_commands = cast("list[dict[str, Any]]", commands_req.json())
        return [CommandResource.from_api(command) for command in raw_commands]
