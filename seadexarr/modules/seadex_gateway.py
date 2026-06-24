"""SeaDex API gateway: fetch an entry by AniList id.

``SeaDexGateway`` wraps the SeaDex client with the connection-error handling the
orchestrator relies on (a missing entry and a SeaDex outage both degrade to
``None`` rather than raising).
"""

import logging

import httpx
from seadex import EntryNotFoundError, EntryRecord, SeaDexEntry


class SeaDexGateway:
    """Thin wrapper over the SeaDex client for entry lookups."""

    def __init__(self, *, logger: logging.Logger) -> None:
        """Instantiate the SeaDex API client.

        Args:
            logger (logging.Logger): For the connection-error warning.
        """

        self.logger = logger
        self.seadex = SeaDexEntry()

    def entry(self, al_id: int) -> EntryRecord | None:
        """Get the SeaDex entry for an AniList id, or None.

        A missing entry (``EntryNotFoundError``) and a SeaDex outage
        (``httpx.ConnectError``) both return None so the caller can skip the id
        gracefully; the outage is surfaced as a warning.

        Args:
            al_id (int): AniList ID
        """

        sd_entry = None
        try:
            sd_entry = self.seadex.from_id(al_id)
        except EntryNotFoundError:
            pass
        except httpx.ConnectError:
            self.logger.warning("Could not connect to SeaDex. Website may be down")

        return sd_entry
