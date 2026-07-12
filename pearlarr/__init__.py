"""Pearlarr: grab the best SeaDex-tagged anime releases for Sonarr and Radarr.

The supported interfaces are the CLI (`pearlarr`), the config schema, and the
notification payloads. Every Python import path - including the names
re-exported here - is internal and may change without notice; pin your exact
version if you script against it anyway.
"""

from typing import TYPE_CHECKING

from .cli import pearlarr_cli
from .log import setup_logger

# `__version__` is resolved LAZILY (PEP 562): the `pearlarr` entry point
# imports this package only to reach `pearlarr_cli`, so it must not eagerly
# pay the `importlib.metadata` lookup. `from . import __version__` in
# cache.py still resolves through `__getattr__`.
if TYPE_CHECKING:
    __version__: str

__all__ = [
    "pearlarr_cli",
    "setup_logger",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        from importlib.metadata import version

        return version("pearlarr")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
