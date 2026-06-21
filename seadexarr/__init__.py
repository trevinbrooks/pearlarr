from importlib.metadata import version

# Get the version
__version__ = version(__name__)

from .modules import SeaDexRadarr, SeaDexSonarr, seadexarr_cli, setup_logger

__all__ = [
    "SeaDexRadarr",
    "SeaDexSonarr",
    "seadexarr_cli",
    "setup_logger",
]
