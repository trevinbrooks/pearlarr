"""The rich console surface (PR3: diagnostics + the boot cockpit; PR4: the scan
arm; grows into the full renderer).

Diagnostics are position-free; this renderer is the single ambient placement
authority: its :class:`~.breadcrumbs.BreadcrumbFold` instance decides where a
diagnostic lands: indented while a boot section, wait region, or entry block is
open, column 0 otherwise (RUN/ITEM alone stays column-0 — the producers open
entry scopes, so a note between entries sits at the run margin). The boot
events (banner / steps / capstone) drive the :class:`~.boot_region.BootRegion`
— the live spinner and the durable ledger lines moved there from ``boot_view``.
The scan events render through the shared :mod:`.scan_lines` builders at
LOGGER-parity gating, so the console shows exactly what the file logs.

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

from .boot_region import BootRegion
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
from .scan_lines import ScanEvent, render_legacy_lines, scan_event_lines
from .trace import CapturedTrace
from ..console_caps import CapsCache, console_of
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
    """The hub's console seat (PR3: diagnostics + boot; scan/wait arms land in PR4-5)."""

    def __init__(
        self,
        console_source: Callable[[], Console | None] = live_console,
        caps_cache: CapsCache | None = None,
    ) -> None:
        # ``caps_cache`` must be the instance the LegacyRenderer echo shares in
        # production (cli wiring); None builds the BootRegion a private cache.
        self._console_source = console_source
        self._crumbs = BreadcrumbFold()
        self._level = int(Severity.INFO)
        self._boot = BootRegion(console_source, caps_cache, level_source=self._current_level)

    def handle(self, event: Event, when: float) -> None:
        del when
        # Placement must be settled BEFORE rendering (fold-first also keeps the
        # fold advancing when rendering raises); a boot-section departure tears
        # the live slot down no matter which event evicted the node.
        boot_was_open = self._boot_section_open()
        self._crumbs.apply(event)
        if boot_was_open and not self._boot_section_open():
            self._boot.section_left()
        match event:
            case Diagnostic():
                self._diagnostic(event)
            case (
                RunStarted()
                | BootStepStarted()
                | BootStepProgressed()
                | BootStepSlow()
                | BootStepFinished()
                | BootReady()
            ):
                self._boot.handle(event)
            case ScopeOpened() | ScopeClosed():
                # Scope boundaries feed the fold (and the departure check) only.
                pass
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
            case (
                CycleStarted()
                | NextRunScheduled()
                | ReleaseSkipped()
                | GrabFailed()
                | ScanFinished()
                | WaitStarted()
                | WaitProgress()
                | TorrentGraduated()
                | WaitFinished()
                | RunFinished()
            ):
                # ReleaseSkipped/GrabFailed producers are still raw logger
                # warnings the bridge adopts; the wait arms arrive with PR5.
                pass
            case _:
                assert_never(event)

    def begin_cycle(self) -> None:
        self._crumbs.reset()
        self._boot.begin_cycle()

    def set_level(self, level: int) -> None:
        self._level = level

    def close(self) -> None:
        self._boot.close()

    def _current_level(self) -> int:
        """The single level store, read live by the boot region's level_source."""

        return self._level

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

    def _scan(self, event: ScanEvent) -> None:
        """The scan console arm: the shared legacy lines over the shared Console.

        Gating is LOGGER parity (``render_legacy_lines``): at a configured
        WARNING the INFO scan lines vanish from the console exactly as they
        vanish from the file — deliberately NOT the diagnostics' console floor.
        """

        console = self._console_source()
        if console is None:
            return
        render_legacy_lines(console, scan_event_lines(event), self._level)

    def _frontier_has(self, *kinds: ScopeKind) -> bool:
        """True when any open node (the whole stack, not just the top) is one of ``kinds``."""

        wanted = frozenset(kinds)
        return any(node.kind in wanted for node in self._crumbs.nodes())

    def _cockpit_open(self) -> bool:
        """True while a boot section, wait region, or entry block is open — the
        indented contexts a diagnostic folds into (RUN/ITEM alone stays column-0)."""

        return self._frontier_has(ScopeKind.BOOT_SECTION, ScopeKind.WAIT_REGION, ScopeKind.ENTRY)

    def _boot_section_open(self) -> bool:
        """Whether the boot section sits on the frontier (the stack is <=3 nodes)."""

        return self._frontier_has(ScopeKind.BOOT_SECTION)
