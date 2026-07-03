from typing import TYPE_CHECKING

from .modules import seadexarr_cli, setup_logger

# ``__version__`` and the run-machinery re-exports are resolved LAZILY (PEP 562):
# the ``seadexarr`` entry point imports this package only to reach
# ``seadexarr_cli``, so it must not eagerly pay the ``importlib.metadata`` lookup
# or pull the heavy run machinery (qBittorrent / arrapi / the SeaDex+httpx chain).
# ``from .. import __version__`` in cache.py and ``seadexarr.<Sync>`` for
# programmatic use both still resolve through ``__getattr__``.
_LAZY_EXPORTS = frozenset({"RadarrSync", "RunDeps", "RunServices", "SeaDexArr", "SonarrSync"})

if TYPE_CHECKING:
    # Declared for type checkers; the values are produced at runtime by
    # __getattr__ (PEP 562), so these never execute / never import eagerly.
    __version__: str
    from .modules import RadarrSync, RunDeps, RunServices, SeaDexArr, SonarrSync

__all__ = [
    "RadarrSync",
    "RunDeps",
    "RunServices",
    "SeaDexArr",
    "SonarrSync",
    "__version__",
    "seadexarr_cli",
    "setup_logger",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        from importlib.metadata import version

        return version("seadexarr")
    if name in _LAZY_EXPORTS:
        from . import modules

        return getattr(modules, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
