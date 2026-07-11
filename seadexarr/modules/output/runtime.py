"""The process-global hub registry — the strangler seam (S3; PR7 revisits ownership).

Mirrors stdlib logging's process-global registry: cli installs the real hub once
pre-loop; scope producers (the boot flow's mark, the wait narrator's factory) and
``apply_log_level`` reach it via :func:`current_hub`. The default is a
renderer-less hub, so emissions before install (tests, library use) drop silently
instead of raising.
"""

from __future__ import annotations

from typing import Final

from .events import Diagnostic, Event, Severity
from .hub import OutputHub, SeverityCounts
from ..log import LOG_NAME

_DEFAULT_HUB: Final = OutputHub([])

_hub: OutputHub = _DEFAULT_HUB


def emit_to_hub(event: Event) -> None:
    """Emit through the process hub, resolved at call time (the strangler seam).

    THE late-resolver emit for every producer without a bound handle - the
    reporter's ``emit`` seam, boot_flow's ledger/mark, the wait narrator's
    ScopeFactory - so the per-cycle hub swap is never captured at build time,
    and the seam has one home.
    """

    current_hub().emit(event)


def hub_note(message: str, *, severity: Severity = Severity.INFO) -> None:
    """A first-party one-liner note through the process hub (the logger.info replacements)."""

    current_hub().emit(Diagnostic(severity=severity, message=message, origin=LOG_NAME))


def hub_counts() -> SeverityCounts:
    """The process hub's severity counts, resolved at call time (emit_to_hub's twin)."""

    return current_hub().counts


def install_hub(hub: OutputHub) -> None:
    """Make ``hub`` the process hub, closing any previously installed one.

    A repeat ``run single`` in one process must not leak the prior hub's open
    FileLogSink (or double-rotate its cascade); the DEFAULT hub is never closed.
    """

    global _hub
    if _hub is not _DEFAULT_HUB and _hub is not hub:
        _hub.close()
    _hub = hub


def uninstall_hub() -> None:
    """Close the installed hub and restore the renderer-less default (tests).

    Closing releases the outgoing hub's sink resources (an open FileLogSink
    handle); emits on a closed hub drop silently. NEVER closes the default.
    """

    global _hub
    if _hub is not _DEFAULT_HUB:
        _hub.close()
    _hub = _DEFAULT_HUB


def current_hub() -> OutputHub:
    """The installed process hub, or the renderer-less default."""

    return _hub
