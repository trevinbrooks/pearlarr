"""The process-global hub registry — the strangler seam (S3; PR7 revisits ownership).

Mirrors stdlib logging's process-global registry: cli installs the real hub once
pre-loop; scope-mark producers (boot/wait views) and ``apply_log_level`` reach it
via :func:`current_hub`. The default is a renderer-less hub, so emissions before
install (tests, library use) drop silently instead of raising.
"""

from __future__ import annotations

from typing import Final

from .hub import OutputHub

_DEFAULT_HUB: Final = OutputHub([])

_hub: OutputHub = _DEFAULT_HUB


def install_hub(hub: OutputHub) -> None:
    """Make ``hub`` the process hub (cli, once; a repeat call re-wires the seam)."""

    global _hub
    _hub = hub


def uninstall_hub() -> None:
    """Restore the renderer-less default (tests)."""

    global _hub
    _hub = _DEFAULT_HUB


def current_hub() -> OutputHub:
    """The installed process hub, or the renderer-less default."""

    return _hub
