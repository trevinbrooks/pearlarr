"""The logging bridge: stdlib LogRecords adopted into Diagnostic events (S5).

One :class:`HubBridgeHandler` per process, attached to the root logger AND to the
app logger (its ``propagate`` stays False, so the app path needs its own seat).
Adoption rules:

* App-logger records: WARNING+ adopt visible (the badge class); sub-WARNING
  adopt ``file_only`` at INFO+ config (DEBUG chatter and unmigrated INFO
  stragglers stay forensic) and under a rich seat (RichConsoleHandler already
  prints the raw record). At a configured DEBUG with a plain/json seat they
  adopt visible — the bridge is the only console route there.
* Root records (third-party: httpx, urllib3, pydantic, py.warnings, ...):
  WARNING+ adopt visible with ``origin = record.name``; sub-WARNING records below
  the hub's level are never constructed, at-or-above it they adopt ``file_only``
  unless the configured level is DEBUG (at DEBUG the hub is a library record's
  only console route, so today's visibility is preserved; at INFO+ the file
  keeps the forensics and stdout loses library chatter).

The bridge CONSTRUCTS new events and never mutates the LogRecord (caplog
safety). A record fired from inside hub dispatch on the same thread (a renderer
or signal handler logging mid-dispatch) downgrades to file-only adoption
(S5 pin 4 / N2). ``logging.captureWarnings`` is flipped on at install, so
warnings-module output arrives via the "py.warnings" logger.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import final, override

from .events import Diagnostic, Severity
from .hub import OutputHub
from .trace import CapturedTrace
from ..log import (
    LOG_NAME,
    HubBridgeBase,
    clear_console_owner,
    register_console_owner,
)


def is_first_party(origin: str) -> bool:
    """True when a diagnostic's origin is the app logger (or a child of it)."""

    return origin == LOG_NAME or origin.startswith(f"{LOG_NAME}.")


def attributed_message(event: Diagnostic) -> str:
    """The rendered message: third-party diagnostics carry their origin up front."""

    if is_first_party(event.origin):
        return event.message
    return f"{event.origin}: {event.message}"


def _severity_of(levelno: int) -> Severity:
    """Map a stdlib levelno onto the Severity ladder (custom levels floor down)."""

    for severity in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO):
        if levelno >= severity:
            return severity
    return Severity.DEBUG


class _DispatchFlag(threading.local):
    """Per-thread reentrancy marker: True while this thread is inside hub dispatch."""

    active: bool = False


@final
class HubBridgeHandler(HubBridgeBase):
    """Adopts stdlib records into Diagnostic events — constructs, never mutates."""

    def __init__(self, hub: OutputHub) -> None:
        super().__init__(level=logging.NOTSET)
        self._hub = hub
        self._flag = _DispatchFlag()

    @override
    def handle(self, record: logging.LogRecord) -> bool:
        # No filter dance (final class, no filters ever attached) and no handler
        # lock: waiting on the hub lock under a handler lock invites inversion.
        self.emit(record)
        return True

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            diagnostic = self._adopt(record)
            if diagnostic is None:
                return
            if self._flag.active:
                # Fired mid-dispatch on this thread: file-only adoption (N2).
                self._hub.emit(replace(diagnostic, file_only=True))
                return
            self._flag.active = True
            try:
                self._hub.emit(diagnostic)
            finally:
                self._flag.active = False
        except Exception:
            self.handleError(record)

    def _adopt(self, record: logging.LogRecord) -> Diagnostic | None:
        """The adoption table (module docstring); None = the record is dropped."""

        if record.levelno >= logging.WARNING:
            file_only = False
        elif is_first_party(record.name):
            # Visible only at DEBUG config on a plain/json seat (the only console
            # route there); a rich seat already prints the raw record, so visible
            # adoption would double it.
            file_only = self._hub.level > logging.DEBUG or self._hub.console_format == "rich"
        elif record.levelno < self._hub.level:
            # Sub-threshold third-party records aren't constructed/counted.
            return None
        else:
            # At a configured DEBUG the hub is a library record's only console
            # route, so it stays visible; at INFO+ the file alone keeps it.
            file_only = self._hub.level > logging.DEBUG
        exc = record.exc_info[1] if record.exc_info is not None else None
        return Diagnostic(
            severity=_severity_of(record.levelno),
            message=record.getMessage(),
            origin=record.name,
            trace=CapturedTrace.from_exception(exc) if exc is not None else None,
            file_only=file_only,
        )


def install_bridge(hub: OutputHub) -> HubBridgeHandler:
    """Attach ONE bridge to the root and app loggers; flip warnings capture on.

    Idempotent by replacement: any prior bridge is removed first, so a repeat
    install (a fresh hub) never doubles up — and ``setup_logger``'s per-cycle
    handler rebuilds preserve the installed one (``HubBridgeBase``).
    """

    uninstall_bridge()
    bridge = HubBridgeHandler(hub)
    logging.getLogger().addHandler(bridge)
    app_logger = logging.getLogger(LOG_NAME)
    app_logger.addHandler(bridge)
    # setup_logger does this too, but doing it here kills the install-window in
    # which an app record would be adopted twice (app seat + root propagation).
    app_logger.propagate = False
    register_console_owner(hub.console_render_active)
    logging.captureWarnings(True)
    return bridge


def uninstall_bridge() -> None:
    """Detach every installed bridge and release warnings capture (tests, re-wiring)."""

    for logger in (logging.getLogger(), logging.getLogger(LOG_NAME)):
        for handler in list(logger.handlers):
            if isinstance(handler, HubBridgeBase):
                logger.removeHandler(handler)
    clear_console_owner()
    logging.captureWarnings(False)
