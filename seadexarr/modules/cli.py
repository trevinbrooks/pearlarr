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


def _build_shared(
    config: str,
    logger: logging.Logger,
) -> tuple[AppConfig, MappingResolver] | None:
    """Load the config once and build the id-mapping resolver both arrs share.

    This is the composition root for a run. The config is read and template-synced
    a single time and returned so each arr reuses it (one read+sync per run, not
    one per arr); the resolver settings are arr-independent, so it's loaded as
    "sonarr" purely to read them. The resolver downloads, parses and indexes the
    three large mapping sources once and is injected into both SeaDexRadarr and
    SeaDexSonarr, so that work also happens a single time per run.

    Returns ``(app_config, resolver)``, or None - after logging the specific
    cause - when the config is missing/unreadable or a mapping source can't be
    fetched, so the caller skips this run and retries next cycle instead of
    crashing. The failure cause is distinguished so the log says whether the user
    needs to fix their config or a source endpoint was unreachable.
    """

    try:
        app_config = AppConfig.load(config, "sonarr")
    except FileNotFoundError:
        logger.error(
            f"No config file at {config} - a starter template was written; "
            "fill it in and re-run. Skipping this run.",
        )
        return None
    except Exception:
        logger.error(f"Could not load config {config}; skipping this run", exc_info=True)
        return None

    try:
        resolver = MappingResolver(
            cache_time=app_config.cache_time,
            ignore_anilist_ids=app_config.ignore_anilist_ids,
            anime_mappings_cfg=app_config.anime_mappings_cfg,
            anidb_mappings_cfg=app_config.anidb_mappings_cfg,
            anibridge_mappings_cfg=app_config.anibridge_mappings_cfg,
        )
    except Exception:
        logger.error(
            "Could not fetch/parse the id-mapping sources; skipping this run",
            exc_info=True,
        )
        return None

    return app_config, resolver


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

        # Load the config and build the id-mapping resolver once, then share both
        # across the two arrs (one config read + one download/parse per cycle). On
        # failure it logs the cause and returns None, so the cycle is skipped and
        # retried on the next pass rather than crashing.
        shared = _build_shared(config, logger)

        if shared is not None:
            app_config, mappings = shared

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
                    app_config=app_config.for_arr("radarr"),
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
                    app_config=app_config.for_arr("sonarr"),
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

    # Load the config and build the shared id-mapping resolver once for whichever
    # arr(s) run (one config read + one download/parse, reused by both). None
    # (after logging the cause) on a config/source failure, in which case the run
    # is skipped.
    shared = None
    if run_radarr or run_sonarr:
        shared = _build_shared(config, logger)

    if shared is not None:
        app_config, mappings = shared

        if run_radarr:
            sdr = None
            try:
                sdr = SeaDexRadarr(
                    config=config,
                    cache=cache,
                    logger=logger,
                    mappings=mappings,
                    app_config=app_config.for_arr("radarr"),
                )
                sdr.run(tmdb_id=movie_id, dry_run=dry_run)
            except Exception:
                logger.error("Unexpected error during Radarr run", exc_info=True)
            finally:
                if sdr is not None:
                    sdr.close()

        if run_sonarr:
            sds = None
            try:
                sds = SeaDexSonarr(
                    config=config,
                    cache=cache,
                    logger=logger,
                    mappings=mappings,
                    app_config=app_config.for_arr("sonarr"),
                )
                sds.run(tvdb_id=series_id, dry_run=dry_run)
            except Exception:
                logger.error("Unexpected error during Sonarr run", exc_info=True)
            finally:
                if sds is not None:
                    sds.close()

    # True when the requested run proceeded (or nothing was requested); False when
    # an arr was requested but the shared config/mappings couldn't be built, so a
    # programmatic caller can tell a no-op-on-failure from a real run.
    return shared is not None or not (run_radarr or run_sonarr)


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
