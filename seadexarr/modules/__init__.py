import importlib
from typing import TYPE_CHECKING

from .cli import seadexarr_cli
from .log import setup_logger

# The public surface is the CLI (``seadexarr_cli``). The composition pieces are
# exported for programmatic use: build a ``RunDeps`` (the shared collaborators),
# wrap it in a ``RunServices`` (the per-id services hub), inject both into
# ``SeaDexArr`` (the run loop) and a strategy, then drive ``SeaDexArr.run_sync``
# - this is what ``cli.py`` does (the facades were dropped).
#
# ``RunDeps``/``RunServices``/``SeaDexArr``/``RadarrSync``/``SonarrSync`` are
# exported LAZILY (PEP 562): importing this package - which the ``seadexarr``
# entry point does just to reach ``seadexarr_cli`` - must not pull the heavy run
# machinery (qBittorrent / arrapi / the SeaDex+httpx chain), so the CLI starts
# fast. They import on first attribute access.
_LAZY: dict[str, str] = {
    "RunDeps": ".run_services",
    "RunServices": ".run_services",
    "SeaDexArr": ".seadex_arr",
    "RadarrSync": ".seadex_radarr",
    "SonarrSync": ".seadex_sonarr",
}

if TYPE_CHECKING:
    from .run_services import RunDeps, RunServices
    from .seadex_arr import SeaDexArr
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
    "RunServices",
    "SeaDexArr",
    "SonarrSync",
    "seadexarr_cli",
    "setup_logger",
]
