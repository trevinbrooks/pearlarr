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
