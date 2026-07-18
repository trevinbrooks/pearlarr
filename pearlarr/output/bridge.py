"""The logging bridge: stdlib LogRecords adopted into Diagnostic events.

One `HubBridgeHandler` per process, attached to the root logger AND to the
app logger (its `propagate` stays False, so the app path needs its own seat).
Adoption rules:

* Sub-WARNING records below the hub's level are never constructed, either
  party: no surface would keep them - the file sink thresholds at the same
  configured level.
* App-logger records: WARNING+ adopt visible (the badge class) - a defensive
  arm, since first-party WARNING+ emits hub Diagnostics directly
  (tests/test_logging_ban.py enforces it. log.py's invalid-level critical is
  the one sanctioned raw site). At-level sub-WARNING (DEBUG chatter, which
  stays raw forever) adopts visible - the bridge is a raw record's ONLY
  console route on every seat (RichConsoleHandler stands down whenever a
  bridge is installed), so the renderer's frontier owns the line's placement
  instead of a producer-baked indent.
* Root records (third-party: httpx, urllib3, pydantic, py.warnings, ...):
  WARNING+ adopt visible with `origin = record.name`. At-level sub-WARNING
  adopts `file_only` unless the configured level is DEBUG (at DEBUG the hub is
  a library record's only console route, so the record keeps its console
  visibility.
  At INFO+ the file keeps the forensics and stdout loses library chatter).

The bridge CONSTRUCTS new events and never mutates the LogRecord (caplog
safety). A record fired from inside hub dispatch on the same thread (a renderer
or signal handler logging mid-dispatch, whichever producer entered the drain)
downgrades to file-only adoption - the hub's own
`dispatch_active` baton read decides, so the rule holds for every drain, not
just bridge-entered ones. The hub is resolved through the process registry at
every record (like every other producer), so an `install_hub` swap can never
orphan the bridge onto a closed hub. `logging.captureWarnings` is flipped on
at install, so warnings-module output arrives via the "py.warnings" logger.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import final, override

from .events import Diagnostic, Severity
from .hub import OutputHub
from .runtime import current_hub
from .trace import CapturedTrace
from ..log import (
    LOG_NAME,
    HubBridgeBase,
    clear_console_owner,
    mark_hub_console_owner,
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


@final
class HubBridgeHandler(HubBridgeBase):
    """Adopts stdlib records into Diagnostic events - constructs, never mutates.

    Holds no hub: the process registry resolves one per record, so a hub swap
    re-points the bridge automatically (the pre-install default drops silently).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)

    @override
    def handle(self, record: logging.LogRecord) -> bool:
        # No filter dance (final class, no filters ever attached) and no handler
        # lock: waiting on the hub lock under a handler lock invites inversion.
        self.emit(record)
        return True

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            hub = current_hub()
            diagnostic = self._adopt(record, hub)
            if diagnostic is None:
                return
            if hub.dispatch_active():
                # Fired mid-dispatch on this thread (a renderer or lifecycle body
                # logging): file-only adoption.
                diagnostic = replace(diagnostic, file_only=True)
            hub.emit(diagnostic)
        except Exception:
            self.handleError(record)

    def _adopt(self, record: logging.LogRecord, hub: OutputHub) -> Diagnostic | None:
        """The adoption table (module docstring). None means the record is dropped."""

        if record.levelno >= logging.WARNING:
            file_only = False
        elif record.levelno < hub.level:
            # sub-threshold: dropped
            return None
        elif is_first_party(record.name):
            # first-party: visible
            file_only = False
        else:
            # third-party: file-only unless DEBUG
            file_only = hub.level > logging.DEBUG
        exc = record.exc_info[1] if record.exc_info is not None else None
        return Diagnostic(
            severity=_severity_of(record.levelno),
            message=record.getMessage(),
            origin=record.name,
            trace=CapturedTrace.from_exception(exc) if exc is not None else None,
            file_only=file_only,
        )


def install_bridge() -> HubBridgeHandler:
    """Attach ONE bridge to the root and app loggers. Flip warnings capture on.

    The bridge feeds whatever hub the registry holds, so a later `install_hub`
    swap re-points it automatically - but install a real hub before (or with)
    the bridge: under the renderer-less default hub, adopted records drop while
    the rich handler stands down. Idempotent by replacement: any prior bridge is removed first, so a
    repeat install never doubles up - and `setup_logger`'s per-cycle handler
    rebuilds preserve the installed one (`HubBridgeBase`).
    """

    uninstall_bridge()
    bridge = HubBridgeHandler()
    logging.getLogger().addHandler(bridge)
    app_logger = logging.getLogger(LOG_NAME)
    app_logger.addHandler(bridge)
    # setup_logger does this too, but doing it here kills the install-window in
    # which an app record would be adopted twice (app seat + root propagation).
    app_logger.propagate = False
    mark_hub_console_owner()
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
