"""The rich console surface (PR2: the diagnostic path; grows into the full renderer).

Diagnostics are position-free; this renderer is the single ambient placement
authority: its :class:`~.breadcrumbs.BreadcrumbFold` instance decides where a
diagnostic lands. In PR2 only the boot/wait cockpit scopes are emitted (the B1/B2
grafts), so placement is: boot-ledger indent while the boot section is open, the
wait indent while the wait region is open, column 0 otherwise. Mid-scan
diagnostics stay column-0 until PR4 emits the scan scopes (bug parity with
today's punch-through).

During the strangler window the renderer resolves the CURRENT shared Console at
render time from the live :class:`~..log.RichConsoleHandler` (``setup_logger``
rebuilds handlers per cycle; the logger identity is stable, S3). Printing
through that shared Console keeps Live reflow safe — the same mechanism as the
cockpits' graduation lines. Under plain/json (no rich handler) it no-ops; the
LegacyRenderer/file path still carries the record.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import assert_never, final

from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

from .breadcrumbs import BreadcrumbFold
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
    ScopeKind,
    ScopeOpened,
    Severity,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)
from .trace import CapturedTrace
from ..console_caps import console_of
from ..log import INDENT, LOG_NAME, RichConsoleHandler, badge_line, console_level


def live_console() -> Console | None:
    """The current cycle's shared rich Console, or None under plain/json."""

    return console_of(logging.getLogger(LOG_NAME))


def diagnostic_threshold(level: int, *, first_party: bool) -> int:
    """The console floor for a diagnostic (S4).

    First-party keeps the ``console_level`` semantics (INFO floor except
    DEBUG/CRITICAL); third-party floors at WARNING unless the configured level
    is DEBUG, so a chatty library can't flood the ledger.
    """

    base = console_level(level)
    if first_party or level == logging.DEBUG:
        return base
    return max(base, logging.WARNING)


def diagnostic_text(event: Diagnostic, *, indented: bool) -> Text:
    """The rendered console line for a diagnostic — pure, golden-testable.

    WARNING+ get the legacy badge word/styles (one look, no drift); INFO/DEBUG
    render dim, in-context (S4 — the unconfigured-arr note's eventual look).
    """

    indent = INDENT if indented else ""
    message = attributed_message(event)
    if RichConsoleHandler.LEVEL_BADGES.get(int(event.severity)) is None:
        return Text(f"{indent}{message}", style="grey50")
    return Text(indent).append_text(badge_line(int(event.severity), message))


@final
class RichRenderer:
    """The hub's console seat (PR2: diagnostics only; other arms land in PR3-5)."""

    def __init__(self, console_source: Callable[[], Console | None] = live_console) -> None:
        self._console_source = console_source
        self._crumbs = BreadcrumbFold()
        self._level = int(Severity.INFO)

    def handle(self, event: Event, when: float) -> None:
        del when
        try:
            match event:
                case Diagnostic():
                    self._diagnostic(event)
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
                    # No producer emits these yet; their arms arrive with PR3-5.
                    pass
                case _:
                    assert_never(event)
        finally:
            # The fold advances even when rendering raises (placement stays true).
            self._crumbs.apply(event)

    def begin_cycle(self) -> None:
        self._crumbs.reset()

    def set_level(self, level: int) -> None:
        self._level = level

    def close(self) -> None:
        pass

    def _diagnostic(self, event: Diagnostic) -> None:
        if event.file_only:
            return
        if int(event.severity) < diagnostic_threshold(self._level, first_party=is_first_party(event.origin)):
            return
        console = self._console_source()
        if console is None:
            return
        # Messages print literally (highlight=False, no markup): "[1/182]" stays text.
        console.print(diagnostic_text(event, indented=self._cockpit_open()), highlight=False, soft_wrap=True)
        if event.trace is not None:
            console.print(
                Traceback(
                    trace=event.trace.rich_trace,
                    show_locals=False,
                    max_frames=CapturedTrace.MAX_FRAMES,
                ),
            )

    def _cockpit_open(self) -> bool:
        """True while a boot section or wait region is open (the two PR2 scopes)."""

        return any(node.kind in (ScopeKind.BOOT_SECTION, ScopeKind.WAIT_REGION) for node in self._crumbs.nodes())
