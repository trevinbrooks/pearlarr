"""The rich console surface: diagnostics + the boot, scan, and wait cockpit arms.

Diagnostics are position-free; this renderer is the single ambient placement
authority: its `breadcrumbs.BreadcrumbFold` instance decides where a
diagnostic lands: indented while a boot section, wait region, or entry block is
open, column 0 otherwise (RUN/ITEM alone stays column-0 — the producers open
entry scopes, so a note between entries sits at the run margin). The boot
events (banner / steps / capstone) drive the `boot_region.BootRegion`
(the live spinner and the durable ledger lines).
The scan events render through the shared `scan_lines` builders at
LOGGER-parity gating, so the console shows exactly what the file logs.

The renderer resolves the CURRENT shared Console at render time from the live
`log.RichConsoleHandler` (`setup_logger` rebuilds handlers per
cycle; the logger identity is stable). Printing through that shared Console
keeps Live reflow safe — the same mechanism as the cockpits' graduation lines.
Under plain/json (no rich handler) it no-ops; this seat is never built there
(cli seats LineRenderer/JsonRenderer instead), and the FileLogSink always
carries the record.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, assert_never, final

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
from .live_region import LiveRegion
from .scan_lines import LegacyLine, ScanEvent, render_legacy_lines, scan_event_lines
from .trace import CapturedTrace
from .wait_region import WaitRegion
from ..console_caps import CapsCache, console_of
from ..log import INDENT, LEVEL_BADGES, LOG_NAME, badge_line, console_level, print_literal


def live_console() -> Console | None:
    """The current cycle's shared rich Console, or None under plain/json."""

    return console_of(logging.getLogger(LOG_NAME))


def diagnostic_threshold(level: int, *, first_party: bool) -> int:
    """The console floor for a diagnostic.

    First-party keeps the `console_level` semantics (INFO floor except
    DEBUG/CRITICAL); third-party floors at WARNING unless the configured level
    is DEBUG, so a chatty library can't flood the ledger.
    """

    base = console_level(level)
    if first_party or level == logging.DEBUG:
        return base
    return max(base, logging.WARNING)


def diagnostic_text(event: Diagnostic, *, indented: bool, use_unicode: bool) -> Text:
    """The rendered console line for a diagnostic — pure, golden-testable.

    WARNING+ get the badge (glyph, or the padded word when `use_unicode` is
    off — one look, no drift); INFO/DEBUG render dim, in-context (the
    unconfigured-arr note's eventual look).
    """

    indent = INDENT if indented else ""
    message = attributed_message(event)
    if LEVEL_BADGES.get(int(event.severity)) is None:
        return Text(f"{indent}{message}", style="grey50")
    return Text(indent).append_text(badge_line(int(event.severity), message, use_unicode=use_unicode))


@final
class RichRenderer:
    """The hub's console seat (diagnostics + the boot/scan/wait cockpit arms)."""

    writes_file_only: ClassVar[bool] = False

    def __init__(
        self,
        console_source: Callable[[], Console | None] = live_console,
        caps_cache: CapsCache | None = None,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        # `caps_cache` is the process-shared instance in production (cli
        # wiring, stable across seat swaps); None builds one private cache
        # shared by this seat's surfaces (diagnostics + both regions).
        self._console_source = console_source
        self._caps = caps_cache if caps_cache is not None else CapsCache()
        self._crumbs = BreadcrumbFold()
        self._level = int(Severity.INFO)
        self._boot = BootRegion(console_source, self._caps, level_source=self._current_level)
        self._wait = WaitRegion(
            console_source,
            self._caps,
            level_source=self._current_level,
            time_source=time_source,
        )
        # Each region owns one Live slot; a frontier departure tears its slot down
        # no matter which event evicted the node (ScopeClosed, a RunFinished unwind).
        self._regions: tuple[tuple[ScopeKind, LiveRegion], ...] = (
            (ScopeKind.BOOT_SECTION, self._boot),
            (ScopeKind.WAIT_REGION, self._wait),
        )

    def handle(self, event: Event, when: float) -> None:
        del when
        # Placement must be settled BEFORE rendering (fold-first also keeps the
        # fold advancing when rendering raises); a region departure tears its live
        # slot down no matter which event evicted the node (ScopeClosed, a
        # RunFinished unwind, anything).
        open_before = tuple(self._frontier_has(kind) for kind, _ in self._regions)
        self._crumbs.apply(event)
        for was_open, (kind, region) in zip(open_before, self._regions, strict=True):
            if was_open and not self._frontier_has(kind):
                region.section_left()
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
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | RunSummaryReady()
            ):
                self._scan(event)
            case WaitStarted() | WaitProgress() | TorrentGraduated() | WaitFinished():
                self._wait.handle(event)
            case NextRunScheduled(at=at):
                self._next_run(at)
            case CycleStarted() | ScanFinished() | RunFinished():
                # Pure boundaries: the banner leads each cycle and the summary
                # closes each run, so none draws a console line of its own.
                pass
            case _:
                assert_never(event)

    def begin_cycle(self) -> None:
        self._crumbs.reset()
        self._boot.begin_cycle()
        self._wait.begin_cycle()

    def set_level(self, level: int) -> None:
        self._level = level

    def close(self) -> None:
        self._boot.close()
        self._wait.close()

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
        use_unicode = self._caps.for_console(console).unicode
        print_literal(console, diagnostic_text(event, indented=self._cockpit_open(), use_unicode=use_unicode))
        if event.trace is not None:
            console.print(
                Traceback(
                    trace=event.trace.rich_trace,
                    show_locals=False,
                    max_frames=CapturedTrace.MAX_FRAMES,
                ),
            )

    def _next_run(self, at: datetime) -> None:
        """The scheduled-mode footer, through the shared durable-line route.

        Weekday included: interval_hours can exceed 24, so a bare HH:MM would be
        ambiguous about which day it means.
        """

        console = self._console_source()
        if console is None:
            return
        line = LegacyLine(logging.INFO, f"Next scheduled run at {at:%a %H:%M}", None)
        render_legacy_lines(console, (line,), self._level)

    def _scan(self, event: ScanEvent) -> None:
        """The scan console arm: the shared legacy lines over the shared Console.

        Gating is LOGGER parity (`render_legacy_lines`): at a configured
        WARNING the INFO scan lines vanish from the console exactly as they
        vanish from the file — deliberately NOT the diagnostics' console floor.
        """

        console = self._console_source()
        if console is None:
            return
        render_legacy_lines(console, scan_event_lines(event), self._level)

    def _frontier_has(self, *kinds: ScopeKind) -> bool:
        """True when any open node (the whole stack, not just the top) is one of `kinds`."""

        wanted = frozenset(kinds)
        return any(node.kind in wanted for node in self._crumbs.nodes())

    def _cockpit_open(self) -> bool:
        """True while a boot section, wait region, or entry block is open.

        These are the indented contexts a diagnostic folds into (RUN/ITEM alone
        stays column-0).
        """

        return self._frontier_has(ScopeKind.BOOT_SECTION, ScopeKind.WAIT_REGION, ScopeKind.ENTRY)
