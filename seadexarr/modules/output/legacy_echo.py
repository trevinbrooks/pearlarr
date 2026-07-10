"""The strangler echo: hub-only facts re-enter the legacy logger for file/plain.

Until PR6 the legacy handlers own the file (and plain/json stdout) surfaces, so
anything that exists only as a hub event must travel back through the app logger
to be persisted. Every re-emission carries the ``HUB_EVENT`` mark: the bridge
drops it (loop-proof) and the rich console handler skips it (the hub's renderer
already owns the console); the FileHandler writes it and LogCounter counts it.

Four event families echo here:

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
* The scan surface (PR4) — banners, ledger rows, entry blocks, action blocks,
  the run summary — re-logged as today's EXACT records via the shared
  :mod:`.scan_lines` builders (message + ``CONSOLE_EXTRA`` payload both ride, so
  a record is indistinguishable from the reporter's own). ``ReleaseSkipped`` /
  ``GrabFailed`` stay pass-arms: nothing produces them yet — the skip/failure
  sites emit ``EntryDetail`` lines until PR6 Band D flips them to the typed events.
* The wait surface (PR5, P3) — graduations and the closing tally always echo;
  the digest start line + throttled "still waiting" pulses echo ONLY on a
  non-live console (a live-TTY run's file never carried them). The pulse cadence
  runs on a :class:`~.wait_lines.PulseThrottle` kept in lockstep with the
  WaitRegion's copy, via the shared :mod:`.wait_lines` builders.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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
from .scan_lines import LegacyLine, ScanEvent, scan_event_lines
from .wait_lines import (
    PulseThrottle,
    WaitEvent,
    wait_graduation_line,
    wait_pulse_line,
    wait_start_line,
    wait_tally_lines,
)
from ..console_caps import Capabilities, CapsCache, console_of
from ..log import CONSOLE_EXTRA, HUB_EVENT, HUB_FILE_ONLY, LOG_NAME


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
        # The non-TTY digest cadence; kept in lockstep with the WaitRegion's copy.
        self._pulse = PulseThrottle()

    def handle(self, event: Event, when: float) -> None:
        del when
        match event:
            case Diagnostic():
                self._echo(event)
            case RunStarted() | BootStepSlow() | BootStepFinished() | BootReady():
                self._boot_ledger(event)
            case (
                ScanStarted()
                | ItemStarted()
                | EntryHeader()
                | EntryDetail()
                | LedgerRow()
                | GrabAction()
                | CapReached()
                | RunSummaryReady()
            ):
                self._scan(event)
            case WaitStarted() | WaitProgress() | TorrentGraduated() | WaitFinished():
                self._wait(event)
            case (
                CycleStarted()
                | NextRunScheduled()
                | ScopeOpened()
                | ScopeClosed()
                | BootStepStarted()
                | BootStepProgressed()
                | ReleaseSkipped()
                | GrabFailed()
                | ScanFinished()
                | RunFinished()
            ):
                # The legacy producers still render these surfaces themselves
                # (ScanFinished/RunFinished have no legacy line at all).
                pass
            case _:
                assert_never(event)

    def begin_cycle(self) -> None:
        self._caps_cache.reset()
        self._pulse.reset()

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

    def _scan(self, event: ScanEvent) -> None:
        """Re-emit a scan event's legacy lines through the app logger.

        Each line carries its exact message AND its ``CONSOLE_EXTRA`` payload,
        so the record is indistinguishable from the reporter's own on every
        legacy surface (file bytes, plain/json stdout, LogCounter).
        """

        # Gate once before assembly: an EntryDetail line carries its own
        # severity; every other scan line is INFO. logger.log re-gates per line.
        gate = int(event.severity) if isinstance(event, EntryDetail) else logging.INFO
        if not self._logger.isEnabledFor(gate):
            return
        self._echo_lines(scan_event_lines(event))

    def _wait(self, event: WaitEvent) -> None:
        """Echo the wait surface (P3): start/pulse only when NOT live, graduations
        and the tally always. The throttle mirrors the WaitRegion's copy so the
        cadence can't drift; fire() advances even when the pulse is level-gated."""

        match event:
            case WaitStarted():
                self._pulse.arm(event.pulse_s)
                # Parity: the live cockpit never logged the start line (the spinner
                # showed liveness); only the non-TTY digest did. Gate before caps.
                if self._logger.isEnabledFor(logging.INFO) and not self._caps().live:
                    self._echo_lines([wait_start_line(event)])
            case WaitProgress(snapshot=snapshot):
                # fire() must advance regardless of level; only the non-TTY digest
                # pulses (a live console shows liveness through the cockpit).
                if not self._caps().live and self._pulse.fire(snapshot.elapsed_s):
                    self._echo_lines([wait_pulse_line(snapshot)])
            case TorrentGraduated():
                if self._logger.isEnabledFor(logging.INFO):
                    self._echo_lines([wait_graduation_line(event, self._caps())])
            case WaitFinished():
                if self._logger.isEnabledFor(logging.INFO):
                    self._echo_lines(wait_tally_lines(event))
            case _:
                assert_never(event)

    def _echo_lines(self, lines: Iterable[LegacyLine]) -> None:
        """Re-emit shared legacy lines through the app logger (HUB_EVENT-marked).

        Each line carries its exact message AND its ``CONSOLE_EXTRA`` payload, so
        the record is indistinguishable from a producer's own on every legacy
        surface (file bytes, plain/json stdout, LogCounter). logger.log re-gates
        per line, so a raised level still drops the INFO lines.
        """

        for line in lines:
            extra: dict[str, object] = {HUB_EVENT: True}
            if line.payload is not None:
                extra[CONSOLE_EXTRA] = line.payload
            self._logger.log(line.level, line.message, extra=extra)

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
