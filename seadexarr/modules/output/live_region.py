"""The shared spine of the renderer's live-slot regions (boot + wait).

:class:`LiveRegion` owns what :class:`~.boot_region.BootRegion` and
:class:`~.wait_region.WaitRegion` had duplicated: the console/caps/level
wiring, the single ``rich.Live`` + spinner slot, and the teardown routes
(frontier departure, cycle boundary, close). Subclasses own their event
handling and extend :meth:`_reset` with per-cycle frame state of their own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from ..console_caps import CapsCache
from ..log import LOG_NAME


class LiveRegion:
    """One live slot + durable prints over the shared Console."""

    def __init__(
        self,
        console_source: Callable[[], Console | None],
        caps_cache: CapsCache | None,
        *,
        level_source: Callable[[], int],
    ) -> None:
        self._console_source = console_source
        # Production wiring (cli) shares ONE cache with the LegacyRenderer echo so
        # both surfaces branch on the same probe; None builds a private cache.
        self._caps_cache = caps_cache if caps_cache is not None else CapsCache()
        # The RichRenderer's level store, read live (no duplicate _level here).
        self._level_source = level_source
        # For contained teardown/render failures; never above DEBUG (see pr5_plan P4).
        self._logger = logging.getLogger(LOG_NAME)
        self._live: Live | None = None
        self._spinner: Spinner | None = None

    def section_left(self) -> None:
        """This region left the renderer's frontier: tear the live slot down.

        A safe no-op when no Live ever started, so the generalized
        frontier-departure loop can call it unconditionally.
        """

        self._stop_live()

    def begin_cycle(self) -> None:
        self._reset()
        self._caps_cache.reset()

    def close(self) -> None:
        self._stop_live()

    def _reset(self) -> None:
        """Drop the live slot; subclasses extend with their per-cycle frame state."""

        self._stop_live()

    def _stop_live(self) -> None:
        # Take-and-clear; the stop() raise is contained so a failed stop can't
        # eat the durable print that follows (the boot capstone, the wait tally).
        live, self._live, self._spinner = self._live, None, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                self._logger.debug("live region stop failed", exc_info=True)
