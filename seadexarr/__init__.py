from importlib.metadata import version

# Get the version
__version__ = version(__name__)

from .modules import (
    RadarrSync,
    RunDeps,
    SeaDexArr,
    SonarrSync,
    seadexarr_cli,
    setup_logger,
)

__all__ = [
    "RadarrSync",
    "RunDeps",
    "SeaDexArr",
    "SonarrSync",
    "seadexarr_cli",
    "setup_logger",
]
