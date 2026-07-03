from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError
from typer.models import OptionInfo

from .boot_view import BootView, make_boot_view
from .config import AppConfig, Arr, template_path
from .log import setup_logger
from .manual_import import ImportWaitMode
from .paths import AppPaths, ensure_data_dir, resolve_paths
from .runlock import single_instance_lock

if TYPE_CHECKING:
    from collections.abc import Generator

    # Imported only for annotations - the runtime imports live in the functions
    # that use them, so their deps aren't pulled at CLI module load (see below).
    from .cache import CacheStore
    from .mappings import MappingResolver


def _option(text: str) -> OptionInfo:
    """``typer.Option(help=...)`` with a fully-known signature (typer's own is
    partially unknown under strict: click 8.4 made ParamType generic).

    ``default=...`` marks "no default here" - Annotated params take the signature
    default, exactly as ``typer.Option`` passes it.
    """

    return OptionInfo(default=..., help=text)


# The heavy clients (qBittorrent / arrapi / the SeaDex+httpx chain via cache) are
# imported lazily inside the functions that use them, NOT at module load, so the
# CLI starts and prints its title without paying ~150ms+ for libraries a `--help`
# or a config/cache subcommand never touches. ``ImportWaitMode`` stays eager: it
# rides a typer command signature, which typer resolves at invocation.

seadexarr_cli = typer.Typer(name="seadexarr_cli")
seadexarr_run = typer.Typer(name="run", help="Run SeaDexArr: a scheduled loop or a one-off single run.")
seadexarr_config = typer.Typer(name="config", help="Manage the config file.")
seadexarr_cache = typer.Typer(name="cache", help="Back up, restore, remove or inspect the cache database.")

seadexarr_cli.add_typer(seadexarr_run)
seadexarr_cli.add_typer(seadexarr_config)
seadexarr_cli.add_typer(seadexarr_cache)


def _remove_db_sidecars(db_path: str) -> None:
    """Remove a SQLite db's WAL/SHM sidecar files if present (best-effort)."""

    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            os.remove(db_path + suffix)


def _refused_by_active_run(acquired: bool, data_dir: str) -> bool:
    """True (after echoing why) when another run holds the single-instance lock.

    Modifying cache.db while a run is live would clobber the in-flight database,
    so the cache commands take the same lock the runner uses and refuse instead.
    """

    if acquired:
        return False
    typer.echo(
        f"Another SeaDexArr run is active in {data_dir}; refusing to modify the cache.",
    )
    return True


@contextlib.contextmanager
def _open_cache_readonly() -> Generator[CacheStore]:
    """Open cache.db read-only for a diagnostic command, closing it afterwards.

    Read-only (no descriptor re-stamp, no WAL switch, no fail-open quarantine) so
    the diagnostic reflects the file as-is; a corrupt/not-a-database file raises
    ``sqlite3.DatabaseError`` from the first read, for the command to report.
    """

    from .cache import CacheStore

    paths = resolve_paths()
    if not os.path.exists(paths.cache):
        raise FileNotFoundError(f"File {paths.cache} not found")

    store = CacheStore.open_readonly(paths.cache)
    try:
        yield store
    finally:
        store.close()


def _build_shared(
    config: str,
    logger: logging.Logger,
    mappings_db: str,
    boot: BootView,
) -> tuple[AppConfig, MappingResolver] | None:
    """Load the config once and build the id-mapping resolver both arrs share.

    The config is read and validated a single time and returned so each arr
    reuses it (one read+sync per run, not one per arr); the resolver settings are
    arr-independent, so it's loaded as "sonarr" purely to read them. The resolver
    downloads-if-stale and (only when a source's content changed) parses+indexes
    the three large mapping sources into ``mappings.db``, then serves both arrs
    from SQL; it is injected (by ``_run_arrs``) into both, so that work happens a
    single time per run and is skipped entirely when the sources are unchanged.

    Returns ``(app_config, resolver)``, or None - after logging the specific
    cause - when the config is missing/unreadable or a mapping source can't be
    fetched, so the caller skips this run and retries next cycle instead of
    crashing. The failure cause is distinguished so the log says whether the user
    needs to fix their config or a source endpoint was unreachable.
    """

    from .mappings import MappingResolver

    try:
        with boot.step("Reading config"):
            app_config = AppConfig.load(config)
    except FileNotFoundError:
        logger.error(
            f"No config file at {config} - a starter template was written; fill it in and re-run. Skipping this run.",
        )
        return None
    except ValidationError as e:
        # Surface the specific bad keys (nested path -> message) without a traceback,
        # then skip + retry next cycle - same contract as the missing-file branch.
        details = "\n".join(f"  - {'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in e.errors())
        logger.error(
            f"Invalid configuration in {config}:\n{details}\nFix the listed keys and re-run. Skipping this run.",
        )
        return None
    except Exception:
        logger.error(f"Could not load config {config}; skipping this run", exc_info=True)
        return None

    try:
        with boot.step("Refreshing mappings") as mapping_step:
            resolver = MappingResolver(
                cache_time=app_config.advanced.cache_time,
                ignore_anilist_ids=app_config.seadex.ignore_anilist_ids,
                anime_mappings_cfg=app_config.mappings.anime_mappings,
                anidb_mappings_cfg=app_config.mappings.anidb_mappings,
                anibridge_mappings_cfg=app_config.mappings.anibridge_mappings,
                mappings_db=mappings_db,
                logger=logger,
                progress=mapping_step,
            )
            # Overwrite the per-MB download detail with the final "which sources"
            # note, so the graduated line reads e.g. "anime-ids · anidb · anibridge".
            mapping_step.note(resolver.sources_summary())
    except Exception:
        logger.error(
            "Could not fetch/parse the id-mapping sources; skipping this run",
            exc_info=True,
        )
        return None

    return app_config, resolver


def _run_arrs(
    arrs: list[tuple[Arr, int | None]],
    *,
    paths: AppPaths,
    logger: logging.Logger,
    dry_run: bool = False,
    import_wait_mode: ImportWaitMode | None = None,
) -> bool:
    """Build the shared config + mappings once, then run each requested arr.

    ``arrs`` is a list of ``(arr_name, item_id)`` pairs; each is delegated to
    ``_run_arr`` (which logs and closes independently, so one crashing doesn't ruin
    the other). The shared config read and mapping download/parse happen a single
    time, and only when at least one arr is requested. Returns True when there was
    nothing to do or the run proceeded; False - after ``_build_shared`` logs the
    cause - when an arr was requested but the shared deps couldn't be built, so a
    caller can tell a no-op-on-failure from a real run. ``import_wait_mode`` is the
    resolved CLI override threaded into each arr (None in scheduled mode).
    """

    if not arrs:
        return True

    # Guard against two runs sharing one data directory (cache.db + WAL); SQLite
    # keeps the file safe, but overlapping runs would duplicate work and could race
    # on imports. A different data dir gets its own lock, so intentional parallel
    # instances are still fine.
    with single_instance_lock(paths.data_dir, logger=logger) as acquired:
        if not acquired:
            logger.warning(
                f"Another SeaDexArr run is active in {paths.data_dir}; skipping this run.",
            )
            return False

        # The startup cockpit: an instant brand title, then a live spinner over the
        # pre-scan IO (config, mappings, cache, qBittorrent, library fetch,
        # prefetch). Built from the logger's console so it degrades to a calm log
        # digest on a non-TTY, and closed in the finally so the terminal is always
        # restored even on an early failure.
        boot = make_boot_view(logger)
        boot.banner()
        # Pull the heavy run machinery now - after the instant title, before the
        # cockpit's first step - so this one-time import cost lands in the gap
        # between the banner and the spinner rather than stalling a live step.
        from .seadex_arr import QbitConnectionError, RunDeps, SeaDexArr
        from .seadex_radarr import RadarrSync
        from .seadex_sonarr import SonarrSync

        try:
            # The parsed/indexed mapping cache lives beside cache.db in the data dir.
            shared = _build_shared(paths.config, logger, paths.mappings_db, boot)
            if shared is None:
                return False

            app_config, mappings = shared
            try:
                for arr_name, item_id in arrs:
                    services = None
                    try:
                        deps = RunDeps.build(
                            arr_name,
                            paths.config,
                            paths.cache,
                            logger,
                            mappings=mappings,
                            app_config=app_config,
                            cache_legacy=paths.cache_legacy,
                            boot=boot,
                        )
                        services = SeaDexArr(deps, arr_name)
                        match arr_name:
                            case Arr.SONARR:
                                services.run_sync(
                                    SonarrSync(deps, services),
                                    arr=arr_name,
                                    item_id=item_id,
                                    dry_run=dry_run,
                                    import_wait_mode=import_wait_mode,
                                    boot=boot,
                                )
                            case Arr.RADARR:
                                services.run_sync(
                                    RadarrSync(deps, services),
                                    arr=arr_name,
                                    item_id=item_id,
                                    dry_run=dry_run,
                                    import_wait_mode=import_wait_mode,
                                    boot=boot,
                                )
                    except QbitConnectionError as e:
                        # A user-facing config problem (wrong host/credentials): a clean
                        # one-line message, not a stack trace under "unexpected error".
                        logger.error(str(e))
                    except Exception:
                        logger.error(f"Unexpected error during {arr_name.capitalize()} run", exc_info=True)
                    finally:
                        # Cap this arr's boot section so the next arr opens a fresh
                        # one; a no-op on the happy path (run_sync already ended it
                        # before scanning), the safety net when a step failed.
                        boot.end_section()
                        if services is not None:
                            services.close()
            finally:
                # The resolver owns mappings.db (shared across both arrs); close it once
                # the cycle is done so the connection / WAL handles are released.
                mappings.close()
        finally:
            boot.close()

        return True


# Default command, schedule run
@seadexarr_cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_dir: Annotated[
        str | None,
        _option(
            "Override the data directory holding config, caches and logs "
            "(default: SEADEX_ARR_DATA_DIR or the OS per-user data directory).",
        ),
    ] = None,
) -> None:
    """SeaDexArr: sync the best SeaDex-tagged anime releases into Sonarr and Radarr.

    Without a subcommand, runs in scheduled mode (both arrs, every few hours).

    \f
    Args:
        data_dir: Override the data directory holding config, caches and logs
            (typer exposes this as ``--data-dir``). Defaults to None, which uses
            ``SEADEX_ARR_DATA_DIR`` or the OS-standard per-user data location.
    """

    # The flag is sugar over SEADEX_ARR_DATA_DIR: fold it into the env so every
    # command's resolve_paths() sees it (commands are also called directly in tests,
    # so the env - not ctx.obj - is the single override channel). Flag wins over a
    # pre-set env because it overwrites here, before any subcommand resolves.
    if data_dir is not None:
        os.environ["SEADEX_ARR_DATA_DIR"] = os.path.abspath(data_dir)

    if ctx.invoked_subcommand is None:
        run_scheduled()


@seadexarr_cli.command("paths")
def show_paths() -> bool:
    """Print the resolved data directory and the files within it."""

    paths = resolve_paths()
    typer.echo(f"data_dir:    {paths.data_dir}")
    typer.echo(f"config:      {paths.config}")
    typer.echo(f"cache:       {paths.cache}")
    typer.echo(f"mappings_db: {paths.mappings_db}")
    typer.echo(f"logs:        {paths.log_dir}")
    return True


@seadexarr_run.command("scheduled")
def run_scheduled() -> None:
    """Run both arr modules on a loop (every SCHEDULE_TIME hours, default 6)."""

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    ensure_data_dir(paths)

    # Get how often to run things
    schedule_time = float(os.getenv("SCHEDULE_TIME", "6"))

    while True:
        logger = setup_logger(log_level="INFO", log_dir=paths.log_dir)

        # Build the shared config + id-mapping resolver once and run both arrs
        # (one config read + one download/parse per cycle, reused by both). On a
        # config/source failure _build_shared logs the cause and the cycle is
        # skipped, so it's retried next pass rather than crashing. No ad-hoc
        # preamble here: the branded title (logged by _run_arrs) leads each cycle,
        # so the scheduled path reads the same as a single run.
        _run_arrs(
            [(Arr.RADARR, None), (Arr.SONARR, None)],
            paths=paths,
            logger=logger,
        )

        next_run_time = (datetime.now() + timedelta(hours=schedule_time)).strftime("%H:%M")
        logger.info(f"Next scheduled run at {next_run_time}")

        time.sleep(schedule_time * 3600)


# Single run. The user-facing help lives on the decorator; the docstring below
# (with its Args block) is for API readers and never reaches --help.
@seadexarr_run.command("single", help="Do a single SeaDexArr run for the selected arr modules.")
def run_single(
    radarr: Annotated[bool, _option("Run the Radarr module.")] = False,
    sonarr: Annotated[bool, _option("Run the Sonarr module.")] = False,
    movie_id: Annotated[
        int | None,
        _option("Only process the movie with this TMDB ID (implies --radarr)."),
    ] = None,
    series_id: Annotated[
        int | None,
        _option("Only process the series with this TVDB ID (implies --sonarr)."),
    ] = None,
    dry_run: Annotated[
        bool,
        _option("Simulate the run: no grabs, no cache writes, no notifications."),
    ] = False,
    import_wait_mode: Annotated[
        ImportWaitMode | None,
        _option("Override the configured imports.wait_mode for this run."),
    ] = None,
) -> bool:
    """Do a single SeaDexArr run for the selected arr modules.

    Args:
        radarr: Do a Radarr run? Defaults to False
        sonarr: Do a Sonarr run? Defaults to False
        movie_id: If set, only run Radarr for the movie with this TMDB ID.
            Implies a Radarr run. Defaults to None
        series_id: If set, only run Sonarr for the series with this TVDB ID.
            Implies a Sonarr run. Defaults to None
        dry_run: If set, simulate the run without grabbing torrents, writing
            the cache, or sending notifications. Defaults to False
        import_wait_mode: Override the configured wait-for-completion + Sonarr
            manual-import mode (off/deferred/blocking/hybrid) for this run. When
            unset the config's ``imports.wait_mode`` wins (cli > config > default).
    """

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    ensure_data_dir(paths)

    logger = setup_logger(log_level="INFO", log_dir=paths.log_dir)

    # Passing a movie/series ID implies running that arr.
    arrs: list[tuple[Arr, int | None]] = []
    if radarr or movie_id is not None:
        arrs.append((Arr.RADARR, movie_id))
    if sonarr or series_id is not None:
        arrs.append((Arr.SONARR, series_id))

    # Build the shared config + mappings once (only when an arr was requested) and
    # run each. True when the run proceeded or nothing was requested; False when an
    # arr was requested but the shared config/mappings couldn't be built, so a
    # programmatic caller can tell a no-op-on-failure from a real run.
    return _run_arrs(
        arrs,
        paths=paths,
        logger=logger,
        dry_run=dry_run,
        import_wait_mode=import_wait_mode,
    )


# Config commands
@seadexarr_config.command("init")
def config_init() -> bool:
    """Write a starter config.yml to the data directory.

    The file lands in the resolved data directory (see the paths command);
    override the location with --data-dir or SEADEX_ARR_DATA_DIR.
    """

    paths = resolve_paths()
    ensure_data_dir(paths)

    shutil.copyfile(template_path(), paths.config)

    return True


# Cache commands
@seadexarr_cache.command("backup")
def cache_backup() -> bool:
    """Back up the cache database to cache.backup.db.

    Uses the SQLite online-backup API so a consistent snapshot is taken even if a
    WAL has uncommitted pages, rather than a raw file copy that could miss them.
    """

    paths = resolve_paths()

    if not os.path.exists(paths.cache):
        raise FileNotFoundError(f"File {paths.cache} not found")

    source = sqlite3.connect(paths.cache)
    dest = sqlite3.connect(paths.cache_backup)
    try:
        source.backup(dest)
    except sqlite3.DatabaseError as e:
        # A corrupt/torn source raises here; report it cleanly rather than crashing
        # the backup command on the very file it is meant to preserve.
        typer.echo(f"cache backup failed: {e}")
        return False
    finally:
        dest.close()
        source.close()

    return True


@seadexarr_cache.command("restore")
def cache_restore() -> bool:
    """Restore the cache database from cache.backup.db.

    Replaces cache.db (and clears any stale WAL/SHM sidecars first so the
    restored snapshot isn't shadowed by a leftover WAL).
    """

    paths = resolve_paths()

    if not os.path.exists(paths.cache_backup):
        raise FileNotFoundError(f"File {paths.cache_backup} not found")

    with single_instance_lock(paths.data_dir) as acquired:
        if _refused_by_active_run(acquired, paths.data_dir):
            return False

        _remove_db_sidecars(paths.cache)
        if os.path.exists(paths.cache):
            os.remove(paths.cache)
        shutil.move(paths.cache_backup, paths.cache)

    return True


@seadexarr_cache.command("remove")
def cache_remove() -> bool:
    """Remove the cache database (cache.db and its WAL/SHM sidecars)."""

    paths = resolve_paths()

    if not os.path.exists(paths.cache):
        raise FileNotFoundError(f"File {paths.cache} not found")

    with single_instance_lock(paths.data_dir) as acquired:
        if _refused_by_active_run(acquired, paths.data_dir):
            return False

        os.remove(paths.cache)
        _remove_db_sidecars(paths.cache)

    return True


@seadexarr_cache.command("stats")
def cache_stats() -> bool:
    """Print cache health: per-block row counts and on-disk size."""

    with _open_cache_readonly() as store:
        try:
            s = store.stats()
        except sqlite3.DatabaseError as e:
            typer.echo(f"cache stats: unreadable database ({e})")
            return False

    size_mib = s.size_bytes / (1024 * 1024)
    typer.echo(
        f"entries={s.entries}  torrent_hashes={s.torrent_hashes}  "
        f"anilist_meta={s.anilist_meta}  sonarr_parse={s.sonarr_parse}  "
        f"pending_imports={s.pending_imports}  size={size_mib:.2f} MiB",
    )
    return True


@seadexarr_cache.command("check")
def cache_check() -> bool:
    """Run a SQLite integrity check on the cache database and print the result."""

    with _open_cache_readonly() as store:
        try:
            result = store.integrity_check()
        except sqlite3.DatabaseError as e:
            # Reporting bad integrity IS this command's job: a result line, not
            # a traceback.
            typer.echo(f"integrity: {e}")
            return False

    typer.echo(f"integrity: {result}")
    return True
