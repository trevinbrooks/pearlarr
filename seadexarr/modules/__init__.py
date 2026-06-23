from .cli import seadexarr_cli
from .log import setup_logger
from .seadex_arr import RunDeps, SeaDexArr
from .seadex_radarr import RadarrSync
from .seadex_sonarr import SonarrSync

# The public surface is the CLI (``seadexarr_cli``). The composition pieces are
# exported for programmatic use: build a ``RunDeps`` (the shared collaborators),
# inject it into ``SeaDexArr`` (the run machinery) and a strategy, then drive
# ``SeaDexArr.run_sync`` - this is what ``cli.py`` does (the facades were dropped).
__all__ = [
    "RadarrSync",
    "RunDeps",
    "SeaDexArr",
    "SonarrSync",
    "seadexarr_cli",
    "setup_logger",
]
