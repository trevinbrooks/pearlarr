import logging
import os
import shutil
import time
from datetime import datetime, timedelta

import typer

from .config import AppConfig
from .mappings import MappingResolver
from .seadex_arr import setup_logger
from .seadex_radarr import SeaDexRadarr
from .seadex_sonarr import SeaDexSonarr

seadexarr_cli = typer.Typer(name="seadexarr_cli")
seadexarr_run = typer.Typer(name="run")
seadexarr_config = typer.Typer(name="config")
seadexarr_cache = typer.Typer(name="cache")

seadexarr_cli.add_typer(seadexarr_run)
seadexarr_cli.add_typer(seadexarr_config)
seadexarr_cli.add_typer(seadexarr_cache)


def _build_shared_resolver(
    config: str,
    logger: logging.Logger,
) -> MappingResolver | None:
    """Build the id-mapping resolver both arrs share for one run.

    The resolver downloads, parses and indexes the three large mapping sources;
    building it once (here, the composition root) and injecting it into both
    SeaDexRadarr and SeaDexSonarr means that work happens a single time per run
    rather than once per arr. The resolver settings are arr-independent, so the
    config is loaded as "sonarr" purely to read them.

    Returns None (after logging) when the config can't be loaded or a source
    can't be fetched, so the scheduled loop skips the cycle and retries next
    time instead of crashing - mirroring the per-arr error handling this
    replaces.
    """

    try:
        config_obj = AppConfig.load(config, "sonarr")
        return MappingResolver(
            cache_time=config_obj.cache_time,
            ignore_anilist_ids=config_obj.ignore_anilist_ids,
            anime_mappings_cfg=config_obj.anime_mappings_cfg,
            anidb_mappings_cfg=config_obj.anidb_mappings_cfg,
            anibridge_mappings_cfg=config_obj.anibridge_mappings_cfg,
        )
    except Exception:
        logger.error("Unexpected error building shared mappings", exc_info=True)
        return None


# Default command, schedule run
@seadexarr_cli.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> bool:
    """Run SeaDexArr in scheduled mode

    Will run both Radarr and Sonarr modules
    """

    if ctx.invoked_subcommand is None:
        run_scheduled()

    return True


@seadexarr_run.command("scheduled")
def run_scheduled() -> None:
    """Run SeaDexArr in scheduled mode

    Will run both Radarr and Sonarr modules
    """

    # Set up config file location
    config_dir = os.getenv("CONFIG_DIR", os.getcwd())
    config = os.path.join(config_dir, "config.yml")
    cache = os.path.join(config_dir, "cache.json")

    # Get how often to run things
    schedule_time = float(os.getenv("SCHEDULE_TIME", "6"))

    while True:

        logger = setup_logger(log_level="INFO")
        logger.info("Starting SeaDexArr in scheduled mode")

        present_time = datetime.now().strftime("%H:%M")
        logger.info(f"Time is {present_time}. Starting scheduled run")

        # Build the id-mapping resolver once and share it across both arrs (one
        # download/parse per cycle). On failure it logs and returns None, so the
        # cycle is skipped and retried on the next pass rather than crashing.
        mappings = _build_shared_resolver(config, logger)

        if mappings is not None:

            # Run both Radarr and Sonarr syncs, catching
            # errors if they do arise. Split them up
            # so one crashing doesn't ruin the other
            sdr = None
            try:
                sdr = SeaDexRadarr(
                    config=config,
                    cache=cache,
                    logger=logger,
                    mappings=mappings,
                )
                sdr.run()
            except Exception:
                logger.error("Unexpected error during Radarr run", exc_info=True)
            finally:
                if sdr is not None:
                    sdr.close()

            sds = None
            try:
                sds = SeaDexSonarr(
                    config=config,
                    cache=cache,
                    logger=logger,
                    mappings=mappings,
                )
                sds.run()
            except Exception:
                logger.error("Unexpected error during Sonarr run", exc_info=True)
            finally:
                if sds is not None:
                    sds.close()

        next_run_time = datetime.now() + timedelta(hours=schedule_time)
        next_run_time = next_run_time.strftime("%H:%M")
        logger.info(f"Scheduled run complete - next run at {next_run_time}")

        time.sleep(schedule_time * 3600)


# Single run
@seadexarr_run.command("single")
def run_single(
    radarr: bool = False,
    sonarr: bool = False,
    movie_id: int | None = None,
    series_id: int | None = None,
    dry_run: bool = False,
) -> bool:
    """Do a single SeaDexArr run

    Args:
        radarr: Do a Radarr run? Defaults to False
        sonarr: Do a Sonarr run? Defaults to False
        movie_id: If set, only run Radarr for the movie with this TMDB ID.
            Implies a Radarr run. Defaults to None
        series_id: If set, only run Sonarr for the series with this TVDB ID.
            Implies a Sonarr run. Defaults to None
        dry_run: If set, simulate the run without grabbing torrents, writing
            the cache, or sending notifications. Defaults to False
    """

    # Set up config file location
    config_dir = os.getenv("CONFIG_DIR", os.getcwd())
    config = os.path.join(config_dir, "config.yml")
    cache = os.path.join(config_dir, "cache.json")

    logger = setup_logger(log_level="INFO")

    # Passing a movie/series ID implies running that arr
    run_radarr = radarr or movie_id is not None
    run_sonarr = sonarr or series_id is not None

    # Build the shared id-mapping resolver once for whichever arr(s) run (one
    # download/parse, reused by both). None (after logging) on a config/source
    # failure, in which case the run is skipped.
    mappings = None
    if run_radarr or run_sonarr:
        mappings = _build_shared_resolver(config, logger)

    if mappings is not None and run_radarr:
        sdr = None
        try:
            sdr = SeaDexRadarr(
                config=config,
                cache=cache,
                logger=logger,
                mappings=mappings,
            )
            sdr.run(tmdb_id=movie_id, dry_run=dry_run)
        except Exception:
            logger.error("Unexpected error during Radarr run", exc_info=True)
        finally:
            if sdr is not None:
                sdr.close()

    if mappings is not None and run_sonarr:
        sds = None
        try:
            sds = SeaDexSonarr(
                config=config,
                cache=cache,
                logger=logger,
                mappings=mappings,
            )
            sds.run(tvdb_id=series_id, dry_run=dry_run)
        except Exception:
            logger.error("Unexpected error during Sonarr run", exc_info=True)
        finally:
            if sds is not None:
                sds.close()

    return True


# Config commands
@seadexarr_config.command("init")
def config_init() -> bool:
    """Initialise a configuration file.

    If not running in Docker, will create a config.yml in the current working
    directory. For Docker, will create config.yml in the /config directory
    """

    config_template_path = os.path.join(os.path.dirname(__file__), "config_sample.yml")

    config_dir = os.environ.get("CONFIG_DIR", os.getcwd())
    config = os.path.join(config_dir, "config.yml")

    shutil.copyfile(config_template_path, config)

    return True


# Cache commands
@seadexarr_cache.command("backup")
def cache_backup() -> bool:
    """Backup cache file.

    Will rename cache to cache.backup.json
    """

    config_dir = os.environ.get("CONFIG_DIR", os.getcwd())
    cache = os.path.join(config_dir, "cache.json")
    backup_cache = os.path.join(config_dir, "cache.backup.json")

    shutil.copyfile(cache, backup_cache)

    return True


@seadexarr_cache.command("restore")
def cache_restore() -> bool:
    """Restore cache file.

    Will rename cache.backup.json to cache.json
    """

    config_dir = os.environ.get("CONFIG_DIR", os.getcwd())
    cache = os.path.join(config_dir, "cache.json")
    backup_cache = os.path.join(config_dir, "cache.backup.json")

    if os.path.exists(backup_cache):
        shutil.move(backup_cache, cache)
    else:
        raise FileNotFoundError(f"File {backup_cache} not found")

    return True


@seadexarr_cache.command("remove")
def cache_remove() -> bool:
    """Remove cache file.

    Will remove cache.json
    """

    config_dir = os.environ.get("CONFIG_DIR", os.getcwd())
    cache = os.path.join(config_dir, "cache.json")

    if os.path.exists(cache):
        os.remove(cache)
    else:
        raise FileNotFoundError(f"File {cache} not found")

    return True
