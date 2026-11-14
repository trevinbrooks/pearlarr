"""The live console's renderer: diagnostics + the boot, scan, and wait branches.

Diagnostics are position-free. This renderer is the single ambient position
authority: its `scope_path.OpenScopes` instance decides where a
diagnostic lands: indented while the boot scope, an item, a wait region, or a title
block is open, column 0 under ARR_RUN alone (before the first item and after the
scan closes. Mid-scan an item stays open until the next boundary,
so import-flow notes indent with the listing they land in). The boot
events (banner / steps / ready line) drive the `boot_region.BootRegion`
(the live spinner and the durable boot step lines).
The scan events render through the shared `scan_lines` builders at
LOGGER-parity gating, so the console shows exactly what the file logs.

The renderer resolves the CURRENT shared Console at render time from the live
`log.RichConsoleHandler` (`setup_logger` rebuilds handlers per
run. The logger identity is stable). Printing through that shared Console
keeps Live reflow safe - the same mechanism as the live regions' durable lines.
Under plain/json (no rich handler) it no-ops. This renderer is never built there
(cli installs PlainRenderer/JsonRenderer instead), and the FileLogRenderer always
carries the record.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import assert_never, final, override

from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

from .boot_region import BootRegion
from .bridge import attributed_message, is_first_party
from .console_lines import ConsoleLine, render_console_lines
from .console_text import INDENT
from .events import (
    ArrRunFinished,
    ArrRunSummaryReady,
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CacheBackedUp,
    CacheIntegrityReported,
    CacheRemoved,
    CacheRestored,
    CacheStatsReported,
    CapReached,
    ConfigMigrated,
    ConfigUpToDate,
    ConfigValidated,
    Diagnostic,
    EffectiveConfigShown,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    NextRunScheduled,
    PathsShown,
    PearlarrStarted,
    ReleaseSkipped,
    RowFinished,
    ScanFinished,
    ScanStarted,
    ScheduledRunStarted,
    ScopeClosed,
    ScopeKind,
    ScopeOpened,
    Severity,
    StarterConfigWritten,
    TitleDetail,
    TitleHeader,
    TitleRow,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)
from .hub import Renderer
from .live_region import LiveRegion
from .scan_lines import ScanEvent, scan_event_lines
from .scope_path import OpenScopes
from .trace import CapturedTrace
from .wait_region import WaitRegion
from ..console_features import DetectedFeatures, console_of
from ..log import LEVEL_BADGES, LOG_NAME, badge_line, console_level, print_literal


def live_console() -> Console | None:
    """The current run's shared rich Console, or None under plain/json."""

    return console_of(logging.getLogger(LOG_NAME))


def diagnostic_threshold(level: int, *, first_party: bool) -> int:
    """The console floor for a diagnostic.

    First-party keeps the `console_level` semantics (INFO floor except
    DEBUG/CRITICAL). Third-party floors at WARNING unless the configured level
    is DEBUG, so a chatty library can't flood the console.
    """

    base = console_level(level)
    if first_party or level == logging.DEBUG:
        return base
    return max(base, logging.WARNING)


def diagnostic_text(event: Diagnostic, *, indented: bool, use_unicode: bool) -> Text:
    """The rendered console line for a diagnostic - pure, golden-testable.

    WARNING+ get the badge (glyph, or the padded word when `use_unicode` is
    off - one look, no drift). INFO/DEBUG render dim, in-context (the
    unconfigured-arr note's eventual look).
    """

    indent = INDENT if indented else ""
    message = attributed_message(event)
    if LEVEL_BADGES.get(int(event.severity)) is None:
        return Text(f"{indent}{message}", style="grey50")
    return Text(indent).append_text(badge_line(int(event.severity), message, use_unicode=use_unicode))


@final
class RichRenderer(Renderer):
    """The hub's console renderer (diagnostics + the boot/scan/wait branches)."""

    def __init__(
        self,
        console_source: Callable[[], Console | None] = live_console,
        detected_features: DetectedFeatures | None = None,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        # `detected_features` is the process-shared instance in production (cli
        # wiring, stable across console renderer swaps). None builds one private instance
        # shared by this renderer's parts (diagnostics + both regions).
        self._console_source = console_source
        self._detected_features = detected_features if detected_features is not None else DetectedFeatures()
        self._open_scopes = OpenScopes()
        self._level = int(Severity.INFO)
        self._boot = BootRegion(console_source, self._detected_features, level_source=self._current_level)
        self._wait = WaitRegion(
            console_source,
            self._detected_features,
            level_source=self._current_level,
            time_source=time_source,
        )
        # Each region owns one Live slot. A scope close tears its slot down
        # no matter which event closed the scope (ScopeClosed, an ArrRunFinished unwind).
        self._regions: tuple[tuple[ScopeKind, LiveRegion], ...] = (
            (ScopeKind.BOOT, self._boot),
            (ScopeKind.WAIT, self._wait),
        )

    @override
    def handle(self, event: Event, when: float) -> None:
        del when
        # Position must be settled BEFORE rendering (applying `OpenScopes` first also keeps
        # it advancing when rendering raises). A region's scope close tears its live
        # slot down no matter which event closed the scope (ScopeClosed, a
        # ArrRunFinished unwind, anything).
        open_before = tuple(self._has_open_scope(kind) for kind, _ in self._regions)
        self._open_scopes.apply(event)
        for was_open, (kind, region) in zip(open_before, self._regions, strict=True):
            if was_open and not self._has_open_scope(kind):
                region.scope_closed()
        match event:
            case Diagnostic():
                self._diagnostic(event)
            case (
                PearlarrStarted()
                | BootStepStarted()
                | BootStepProgressed()
                | BootStepSlow()
                | BootStepFinished()
                | BootReady()
            ):
                self._boot.handle(event)
            case ScopeOpened() | ScopeClosed():
                # Scope boundaries feed `OpenScopes` (and the scope-close check) only.
                pass
            case (
                ScanStarted()
                | ItemStarted()
                | TitleHeader()
                | TitleDetail()
                | TitleRow()
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | ArrRunSummaryReady()
            ):
                self._scan(event)
            case WaitStarted() | WaitProgress() | RowFinished() | WaitFinished():
                self._wait.handle(event)
            case NextRunScheduled(at=at):
                self._next_run(at)
            case ScheduledRunStarted() | ScanFinished() | ArrRunFinished():
                # Pure boundaries: the banner leads each run and the summary
                # closes each arr run, so none draws a console line of its own.
                pass
            case (
                PathsShown()
                | StarterConfigWritten()
                | ConfigValidated()
                | ConfigUpToDate()
                | ConfigMigrated()
                | EffectiveConfigShown()
                | CacheBackedUp()
                | CacheRestored()
                | CacheRemoved()
                | CacheStatsReported()
                | CacheIntegrityReported()
            ):
                # cli command facts - never emitted during a run.
                pass
            case _:
                assert_never(event)

    @override
    def begin_run(self) -> None:
        self._open_scopes.reset()
        self._boot.begin_run()
        self._wait.begin_run()

    @override
    def set_level(self, level: int) -> None:
        self._level = level

    @override
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
        use_unicode = self._detected_features.for_console(console).can_unicode
        print_literal(console, diagnostic_text(event, indented=self._indented_scope_open(), use_unicode=use_unicode))
        if event.trace is not None:
            console.print(
                Traceback(
                    trace=event.trace.rich_trace,
                    show_locals=False,
                    max_frames=CapturedTrace.MAX_FRAMES,
                ),
            )

    def _render_console_lines(self, lines: Iterable[ConsoleLine]) -> None:
        """Shared console lines over the live Console at LOGGER-parity gating."""

        console = self._console_source()
        if console is None:
            return
        render_console_lines(console, lines, self._level)

    def _next_run(self, at: datetime) -> None:
        """The scheduled-mode footer, through the shared durable-line route.

        Weekday included: interval_hours can exceed 24, so a bare HH:MM would be
        ambiguous about which day it means.
        """

        line = ConsoleLine(logging.INFO, f"Next scheduled run at {at:%a %H:%M}", None)
        self._render_console_lines((line,))

    def _scan(self, event: ScanEvent) -> None:
        """The scan console branch: the shared console lines over the shared Console.

        Gating is LOGGER parity (`render_console_lines`): at a configured
        WARNING the INFO scan lines vanish from the console exactly as they
        vanish from the file - deliberately NOT the diagnostics' console floor.
        """

        # Skip building the (sometimes hefty, e.g. the arr run summary) scan lines
        # when there's no console to render them. `_render_console_lines` discards them.
        if self._console_source() is None:
            return
        self._render_console_lines(scan_event_lines(event))

    def _has_open_scope(self, *kinds: ScopeKind) -> bool:
        """True when any open scope (the whole stack, not just the top) is one of `kinds`."""

        wanted = frozenset(kinds)
        return any(open_scope.kind in wanted for open_scope in self._open_scopes.scopes())

    def _indented_scope_open(self) -> bool:
        """True while the boot scope, an item, a wait region, or a title block is open.

        These are the indented contexts a diagnostic goes into (ARR_RUN alone
        stays column-0). ITEM is included so an import-flow warning fired
        between a series' title blocks indents with the listing instead of
        breaking it at the arr run margin.
        """

        return self._has_open_scope(ScopeKind.BOOT, ScopeKind.ITEM, ScopeKind.WAIT, ScopeKind.TITLE)
