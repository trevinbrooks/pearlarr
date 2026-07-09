"""The strangler echo: adopted third-party diagnostics re-enter the legacy logger.

Until PR6 the legacy handlers own the file (and plain/json stdout) surfaces, so a
bridge-adopted third-party record must travel back through the app logger to be
persisted. The re-emission carries the ``HUB_EVENT`` mark: the bridge drops it
(loop-proof) and the rich console handler skips it (the hub's renderer already
owns the console); the FileHandler writes it and LogCounter counts it — so
third-party stragglers now reach the file and the run's issue tally (N1, a
deliberate fix). ``file_only`` diagnostics (hub containment notes) additionally
carry ``HUB_FILE_ONLY``: the file still persists them (the only pre-PR6 sink),
but plain/json stdout and LogCounter skip them. First-party diagnostics are
SKIPPED: their records already traversed the legacy file handler, and echoing
would double-write the file.
"""

from __future__ import annotations

import logging
from typing import assert_never, final

from .bridge import attributed_message, is_first_party
from .events import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CapReached,
    CycleStarted,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    LedgerRow,
    NextRunScheduled,
    ReleaseSkipped,
    RunFinished,
    RunStarted,
    RunSummaryReady,
    ScanFinished,
    ScanStarted,
    ScopeClosed,
    ScopeOpened,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)
from ..log import HUB_EVENT, HUB_FILE_ONLY, LOG_NAME


@final
class LegacyRenderer:
    """Re-emits adopted third-party diagnostics through the app logger (PR2-5)."""

    def handle(self, event: Event, when: float) -> None:
        del when
        match event:
            case Diagnostic():
                self._echo(event)
            case (
                RunStarted()
                | CycleStarted()
                | NextRunScheduled()
                | ScopeOpened()
                | ScopeClosed()
                | BootStepStarted()
                | BootStepProgressed()
                | BootStepSlow()
                | BootStepFinished()
                | BootReady()
                | ScanStarted()
                | ItemStarted()
                | EntryHeader()
                | EntryDetail()
                | LedgerRow()
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | ScanFinished()
                | RunSummaryReady()
                | WaitStarted()
                | WaitProgress()
                | TorrentGraduated()
                | WaitFinished()
                | RunFinished()
            ):
                # The legacy producers still render these surfaces themselves.
                pass
            case _:
                assert_never(event)

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def _echo(event: Diagnostic) -> None:
        if is_first_party(event.origin):
            return
        # The logger admits per configured level (third-party DEBUG/INFO reach
        # the file only when configured); checked first, before string assembly.
        logger = logging.getLogger(LOG_NAME)
        severity = int(event.severity)
        if not logger.isEnabledFor(severity):
            return
        message = attributed_message(event)
        if event.trace is not None:
            # The plain traceback rides the message: the file formatter writes it
            # inline, exactly where a legacy exc_info traceback would land.
            message += "\n" + event.trace.plain_text().rstrip("\n")
        extra: dict[str, bool] = {HUB_EVENT: True}
        if event.file_only:
            extra[HUB_FILE_ONLY] = True
        logger.log(severity, message, extra=extra)
