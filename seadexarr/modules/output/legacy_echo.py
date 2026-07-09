"""The strangler echo: hub-only facts re-enter the legacy logger for file/plain.

Until PR6 the legacy handlers own the file (and plain/json stdout) surfaces, so
anything that exists only as a hub event must travel back through the app logger
to be persisted. Every re-emission carries the ``HUB_EVENT`` mark: the bridge
drops it (loop-proof) and the rich console handler skips it (the hub's renderer
already owns the console); the FileHandler writes it and LogCounter counts it.

Two event families echo here:

* Adopted third-party diagnostics — so stragglers now reach the file and the
  run's issue tally (N1, a deliberate fix). ``file_only`` diagnostics (hub
  containment notes) additionally carry ``HUB_FILE_ONLY``: the file still
  persists them, but plain/json stdout and LogCounter skip them. First-party
  diagnostics are SKIPPED: their records already traversed the legacy file
  handler, and echoing would double-write the file.
* The boot ledger (PR3) — banner, slow heads-up, graduated steps, capstone —
  re-logged as today's EXACT lines (shared builders in :mod:`.boot_region`;
  glyphs follow the same console-caps probe the old view used, so file/plain
  bytes and LogCounter tallies are unchanged). The heads-up echoes only when the
  console is NOT live-capable: the old ``LiveBootView`` never logged it, so a
  live-TTY run's file must not gain it now.
"""

from __future__ import annotations

import logging
from typing import assert_never, final

from .boot_region import banner_title, data_dir_line, graduation_line, ready_line, slow_line
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
from ..console_caps import Capabilities, CapsCache, console_of
from ..log import HUB_EVENT, HUB_FILE_ONLY, LOG_NAME


@final
class LegacyRenderer:
    """Re-emits hub-only events (diagnostics + the boot ledger) through the app
    logger, byte-identical to the pre-hub lines (PR2-5)."""

    def __init__(self, caps_cache: CapsCache | None = None) -> None:
        # Production wiring (cli) shares ONE cache with the BootRegion so both
        # surfaces branch on the same probe; None builds a private cache.
        self._caps_cache = caps_cache if caps_cache is not None else CapsCache()
        # Logger identity is stable across cycles; only its handlers rebuild.
        self._logger = logging.getLogger(LOG_NAME)

    def handle(self, event: Event, when: float) -> None:
        del when
        match event:
            case Diagnostic():
                self._echo(event)
            case RunStarted() | BootStepSlow() | BootStepFinished() | BootReady():
                self._boot_ledger(event)
            case (
                CycleStarted()
                | NextRunScheduled()
                | ScopeOpened()
                | ScopeClosed()
                | BootStepStarted()
                | BootStepProgressed()
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
        self._caps_cache.reset()

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass

    def _boot_ledger(self, event: RunStarted | BootStepSlow | BootStepFinished | BootReady) -> None:
        # Every ledger echo is INFO: gate once, before any caps probe / assembly.
        if not self._logger.isEnabledFor(logging.INFO):
            return
        match event:
            case RunStarted(version=version, data_dir=data_dir):
                # Today's exact three records: title, the blank under it, the data dir.
                self._echo_line(banner_title(version))
                self._echo_line("")
                self._echo_line(data_dir_line(data_dir))
            case BootStepSlow(label=label):
                caps = self._caps()
                # Parity: the old live cockpit never logged the heads-up (the
                # spinner showed liveness); only the log-digest path did.
                if not caps.live:
                    self._echo_line(slow_line(label, caps))
            case BootStepFinished():
                self._echo_line(graduation_line(event, self._caps()))
            case BootReady(elapsed_s=elapsed_s):
                self._echo_line(ready_line(elapsed_s))

    def _echo_line(self, message: str) -> None:
        """One INFO boot-ledger line back through the app logger."""

        self._logger.info(message, extra={HUB_EVENT: True})

    def _caps(self) -> Capabilities:
        """The same probe the old view ran: glyphs follow the logger's console."""

        return self._caps_cache.for_console(console_of(self._logger))

    def _echo(self, event: Diagnostic) -> None:
        if is_first_party(event.origin):
            return
        # The logger admits per configured level (third-party DEBUG/INFO reach
        # the file only when configured); checked first, before string assembly.
        logger = self._logger
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
