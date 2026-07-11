"""The package internals behind the CLI.

The public surface is the CLI (`pearlarr_cli`). The composition pieces are
exported for programmatic use: build a `RunDeps` (the shared collaborators),
wrap it in a `RunServices` (the per-id services hub), inject both into
`RunLoop` and a strategy, then drive `RunLoop.run_sync` - this is what
`bootstrap.py` does.
"""

import importlib
from typing import TYPE_CHECKING

from .cli import pearlarr_cli
from .log import setup_logger

# `RunDeps`/`RunServices`/`RunLoop`/`RadarrSync`/`SonarrSync` are
# exported LAZILY (PEP 562): importing this package - which the `pearlarr`
# entry point does just to reach `pearlarr_cli` - must not pull the heavy run
# machinery (qBittorrent / the SeaDex+httpx chain), so the CLI starts
# fast. They import on first attribute access.
_LAZY: dict[str, str] = {
    "RunDeps": ".run_services",
    "RunServices": ".run_services",
    "RunLoop": ".run_loop",
    "RadarrSync": ".seadex_radarr",
    "SonarrSync": ".seadex_sonarr",
}

if TYPE_CHECKING:
    from .run_loop import RunLoop
    from .run_services import RunDeps, RunServices
    from .seadex_radarr import RadarrSync
    from .seadex_sonarr import SonarrSync


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


__all__ = [
    "RadarrSync",
    "RunDeps",
    "RunLoop",
    "RunServices",
    "SonarrSync",
    "pearlarr_cli",
    "setup_logger",
]
