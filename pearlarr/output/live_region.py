"""The shared base class of the renderer's live regions (boot + wait).

`LiveRegion` owns the console/features/level wiring, the single `rich.Live` +
spinner slot, and the teardown routes (scope close, run boundary,
close) shared by `boot_region.BootRegion` and `wait_region.WaitRegion`.
Subclasses own their event handling and extend `_reset` with per-run frame
state of their own.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from .events import Diagnostic, Severity
from .process_hub import emit_to_hub
from .trace import CapturedTrace
from ..console_features import DetectedFeatures


class LiveRegion:
    """One live slot + durable prints over the shared Console."""

    def __init__(
        self,
        console_source: Callable[[], Console | None],
        detected_features: DetectedFeatures | None,
        *,
        level_source: Callable[[], int],
    ) -> None:
        self._console_source = console_source
        # Production wiring (cli) shares ONE DetectedFeatures across the console renderer's
        # regions so they branch on the same detected features. None builds a private one.
        self._detected_features = detected_features if detected_features is not None else DetectedFeatures()
        # The RichRenderer's level store, read live (no duplicate _level here).
        self._level_source = level_source
        self._live: Live | None = None
        self._spinner: Spinner | None = None

    def scope_closed(self) -> None:
        """This region's scope closed in the renderer's open scopes: tear the live slot down.

        A safe no-op when no Live ever started, so the generalized
        scope-close loop can call it unconditionally.
        """

        self._stop_live()

    def begin_run(self) -> None:
        self._reset()
        self._detected_features.reset()

    def close(self) -> None:
        self._stop_live()

    def _reset(self) -> None:
        """Drop the live slot. Subclasses extend with their per-run frame state."""

        self._stop_live()

    def _stop_live(self) -> None:
        # Take-and-clear. The stop() raise is contained so a failed stop can't
        # eat the durable print that follows (the boot ready line, the wait tally).
        live, self._live, self._spinner = self._live, None, None
        if live is not None:
            try:
                live.stop()
            except Exception as exc:
                self._report_contained("live region stop failed", exc)

    def _report_contained(self, message: str, exc: Exception) -> None:
        """A contained-failure note: a file-only WARNING Diagnostic, main-thread only.

        Straight to the hub, never stdlib logging: a stdlib DEBUG log would die
        at the logger's level gate on any config above DEBUG, leaving a
        persistently broken live region with zero forensics. `file_only` keeps it
        off the console and out of the counts (forensic, like the hub's own
        containment notes). Emitting from inside dispatch is the documented
        re-entrant path (it enqueues under the drain lock). The refresh thread must
        still never call this - `_LiveFrame` latches instead (the ABBA pin).
        """

        emit_to_hub(
            Diagnostic(
                severity=Severity.WARNING,
                message=message,
                origin="output.live_region",
                trace=CapturedTrace.from_exception(exc),
                file_only=True,
            ),
        )
