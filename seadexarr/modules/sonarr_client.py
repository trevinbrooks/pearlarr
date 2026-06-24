"""Sonarr REST client: the HTTP surface the Sonarr syncer talks to.

``SonarrClient`` wraps the high-level ``arrapi`` client (``all_series``)
and the two raw endpoints the syncer needs
(``/api/v3/episode`` and ``/api/v3/parse``)
"""
import logging
from urllib.parse import urlencode

import requests
from arrapi import SonarrAPI

from .log import indent_string
from .seadex_types import SonarrEpisode


class SonarrClient:
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

    def all_series(self) -> list:
        """Every series in Sonarr (unfiltered)."""

        return self._api.all_series()

    def episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """All episodes for a series, season/episode-sorted (``/api/v3/episode``).

        Returns None (with a warning) if Sonarr is unreachable, so the caller can
        skip the id gracefully.

        Args:
            series_id (int): Series ID in Sonarr.
        """

        eps_req_url = (
            f"{self._url}/api/v3/episode?"
            f"seriesId={series_id}&"
            f"includeImages=false&"
            f"includeEpisodeFile=true&"
            f"apikey={self._api_key}"
        )
        eps_req = self._session.get(eps_req_url)

        if eps_req.status_code != 200:
            self._logger.warning(
                "Could not fetch episode data from Sonarr; it may be unreachable",
            )
            return None

        # Sort by season/episode number for slicing later, then parse each raw
        # record into a SonarrEpisode at this client boundary.
        raw_eps = sorted(
            eps_req.json(),
            key=lambda x: (
                x.get("seasonNumber", None),
                x.get("episodeNumber", None),
            ),
        )
        return [SonarrEpisode.from_api(ep) for ep in raw_eps]

    def parse(self, filename: str) -> list:
        """Ask Sonarr to parse a single filename into season/episode numbers.

        Only the season/episode mapping is returned - the file size is filled in
        by the caller, since it comes from the SeaDex file list rather than from
        Sonarr.

        Args:
            filename (str): Filename to parse (basename, not full path).

        Returns:
            list: List of {"season", "episode"} dicts (empty if Sonarr couldn't
                parse the filename).
        """

        d = {"title": filename, "apikey": self._api_key}
        d_enc = urlencode(d)

        # Parse through Sonarr
        parse_req_url = f"{self._url}/api/v3/parse?{d_enc}"
        parse_req = self._session.get(parse_req_url)

        if parse_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not parse {filename} via Sonarr "
                    f"(status code {parse_req.status_code}); skipping file",
                ),
            )
            return []

        episode_info = parse_req.json().get("episodes", [])

        parsed = []
        for ep in episode_info:

            season = ep.get("seasonNumber", None)
            episode = ep.get("episodeNumber", None)

            if season is None or episode is None:
                self._logger.debug(
                    indent_string(
                        f"Season or episode came up None for {filename}; "
                        f"skipping this episode entry",
                    ),
                )
                continue

            parsed.append({"season": season, "episode": episode})

        return parsed

    def manual_import_candidates(
        self,
        *,
        folder: str,
        series_id: int,
        season_number: int | None = None,
        filter_existing_files: bool = False,
    ) -> list[dict]:
        """List Sonarr's manual-import candidates for a folder, series-pinned.

        Passing ``seriesId`` makes Sonarr parse the files *in the context of the
        known series* (PR #7727), which is far more reliable than a blind parse.

        Returns an empty list (with a warning) on a non-200, so the caller can
        leave the import pending and retry later.

        Args:
            folder (str): Folder on disk to scan (URL-encoded into the query).
            series_id (int): Series ID in Sonarr to pin the parse to.
            season_number (int | None): Optional season to scope candidates to.
            filter_existing_files (bool): If True, Sonarr drops files it already
                has imported. Sent lowercase ``true``/``false``.

        Returns:
            list[dict]: Raw ManualImportResource dicts (keys like ``path``,
                ``episodes``, ``quality``, ``languages``, ``releaseGroup``,
                ``rejections``); empty on failure.
        """

        params: dict[str, str] = {
            "folder": folder,
            "seriesId": str(series_id),
            "filterExistingFiles": "true" if filter_existing_files else "false",
        }
        if season_number is not None:
            params["seasonNumber"] = str(season_number)
        params["apikey"] = self._api_key
        params_enc = urlencode(params)

        candidates_req_url = f"{self._url}/api/v3/manualimport?{params_enc}"
        candidates_req = self._session.get(candidates_req_url)

        if candidates_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch manual-import candidates for folder "
                    f"{folder} (status code {candidates_req.status_code}); "
                    f"leaving import pending",
                ),
            )
            return []

        return candidates_req.json()

    def manual_import_execute(
        self,
        *,
        files: list[dict],
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
            files (list[dict]): ManualImport file payloads to import.
            import_mode (str): Sonarr ``importMode``: ``auto`` (default; respects
                the copy/hardlink setting and preserves seeding), ``move`` or
                ``copy``.

        Returns:
            int | None: The queued command's id, or None on failure.
        """

        d = {"apikey": self._api_key}
        d_enc = urlencode(d)

        command_req_url = f"{self._url}/api/v3/command?{d_enc}"
        command_body = {
            "name": "ManualImport",
            "importMode": import_mode,
            "files": files,
        }
        command_req = self._session.post(command_req_url, json=command_body)

        if command_req.status_code not in (200, 201):
            self._logger.warning(
                indent_string(
                    f"Could not queue ManualImport command "
                    f"(status code {command_req.status_code}); "
                    f"leaving import pending",
                ),
            )
            return None

        return command_req.json().get("id")

    def quality_definitions(self) -> list[dict]:
        """All Sonarr quality definitions (``/api/v3/qualitydefinition``).

        Used to resolve a quality NAME (e.g. ``Bluray-2160p``) to a Sonarr
        QualityModel for the manual-import payload.

        Returns an empty list (with a warning) on a non-200, so the caller can
        fall back to other quality sources.

        Returns:
            list[dict]: Raw QualityDefinitionResource dicts; empty on failure.
        """

        defs_req_url = f"{self._url}/api/v3/qualitydefinition?apikey={self._api_key}"
        defs_req = self._session.get(defs_req_url)

        if defs_req.status_code != 200:
            self._logger.warning(
                "Could not fetch quality definitions from Sonarr; "
                "it may be unreachable",
            )
            return []

        return defs_req.json()

    def languages(self) -> list[dict]:
        """All Sonarr languages (``/api/v3/language``).

        Used to resolve language names to ``{id, name}`` objects for the
        manual-import payload.

        Returns an empty list (with a warning) on a non-200, so the caller can
        fall back to the candidate's languages.

        Returns:
            list[dict]: Raw LanguageResource dicts; empty on failure.
        """

        langs_req_url = f"{self._url}/api/v3/language?apikey={self._api_key}"
        langs_req = self._session.get(langs_req_url)

        if langs_req.status_code != 200:
            self._logger.warning(
                "Could not fetch languages from Sonarr; it may be unreachable",
            )
            return []

        return langs_req.json()

    def command_status(self, command_id: int) -> dict:
        """Current state of a Sonarr command (``/api/v3/command/{id}``).

        Used to optionally verify a ``ManualImport`` completed before the caller
        removes the pending record.

        Returns an empty dict (with a warning) on a non-200, so the caller can
        treat the import as unverified and leave it pending.

        Args:
            command_id (int): Command ID returned by ``manual_import_execute``.

        Returns:
            dict: Raw CommandResource dict (keys like ``status``, ``result``);
                empty on failure.
        """

        status_req_url = (
            f"{self._url}/api/v3/command/{command_id}?apikey={self._api_key}"
        )
        status_req = self._session.get(status_req_url)

        if status_req.status_code != 200:
            self._logger.warning(
                indent_string(
                    f"Could not fetch status for command {command_id} "
                    f"(status code {status_req.status_code})",
                ),
            )
            return {}

        return status_req.json()
