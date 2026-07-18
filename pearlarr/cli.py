"""The typer command surface: subcommands, process wiring, and exit codes. `bootstrap.py` owns composition."""

from __future__ import annotations

import contextlib
import itertools
import math
import os
import shutil
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from . import bootstrap
from .config import (
    AppConfig,
    Arr,
    ArrTarget,
    ConfigRewriteError,
    LogFormat,
    strip_userinfo,
    upgrade_config_file,
    write_starter_config,
)
from .console_caps import CapsCache
from .log import LogLevel, resolve_console_format, setup_logger
from .manual_import import ImportWaitMode
from .output import (
    CacheBackedUp,
    CacheIntegrityReported,
    CacheRemoved,
    CacheRestored,
    CacheStatsReported,
    ConfigMigrated,
    ConfigUpToDate,
    ConfigValidated,
    CycleStarted,
    EffectiveConfigShown,
    FileLogSink,
    JsonRenderer,
    JsonValue,
    LineRenderer,
    NextRunScheduled,
    OutputHub,
    PathsShown,
    Renderer,
    StarterConfigWritten,
    emit_to_hub,
    hub_error,
    hub_note,
    hub_warn,
    install_hub,
)
from .output.bridge import install_bridge
from .output.cli_surface import cli_surface
from .output.rich_renderer import RichRenderer
from .paths import PROJECT_URL, AppPaths, ensure_data_dir, resolve_paths
from .runlock import single_instance_lock

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import FrameType

    # Imported only for annotations - the runtime import lives in the function
    # that uses it, so its deps aren't pulled at CLI module load (see below).
    from .cache import CacheStore


# The heavy clients (qBittorrent / the SeaDex+httpx chain via cache) are
# imported lazily inside the functions that use them, NOT at module load, so the
# CLI starts and prints its title without paying ~150ms+ for libraries a `--help`
# or a config/cache subcommand never touches. `ImportWaitMode` stays eager: it
# rides a typer command signature, which typer resolves at invocation.


def _exit_on_failure(result: object, **_: object) -> None:
    """Turn a command's `False` return into exit code 1.

    typer/click ignore return values, so without this every failed command exits
    0 and scripts can't detect the failure. Commands keep returning `bool` for
    programmatic callers (tests call them directly, bypassing this callback).
    The ROOT app's registration is the load-bearing one (a sub-command's False
    propagates up through callback-less groups). The sub-app registrations are
    defense-in-depth, so don't "simplify" the root one away.
    """

    if result is False:
        raise typer.Exit(1)


pearlarr_cli = typer.Typer(
    name="pearlarr",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=f"Docs & issues: {PROJECT_URL}",
    result_callback=_exit_on_failure,
)
pearlarr_run = typer.Typer(
    name="run",
    help="Run Pearlarr: a scheduled loop or a one-off single run.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)
pearlarr_config = typer.Typer(
    name="config",
    help="Initialize, validate, or inspect the config file.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)
pearlarr_cache = typer.Typer(
    name="cache",
    help="Back up, restore, remove, or inspect the cache database.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)

pearlarr_cli.add_typer(pearlarr_run)
pearlarr_cli.add_typer(pearlarr_config)
pearlarr_cli.add_typer(pearlarr_cache)


def _remove_db_sidecars(db_path: str) -> None:
    """Remove a SQLite db's WAL/SHM sidecar files if present (best-effort)."""

    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            os.remove(db_path + suffix)


def _echo_missing(path: str, *, what: str, hint: str) -> bool:
    """True (after echoing why, to stderr) when `path` doesn't exist.

    The cache/config commands report a missing file as one line - naming the
    missing `what` plus a `hint` on how to get one - and a failure exit,
    not a `FileNotFoundError` traceback.
    """

    if os.path.exists(path):
        return False
    hub_error(f"No {what} at {path} - {hint}")
    return True


def _echo_missing_cache(path: str) -> bool:
    """`_echo_missing` for cache.db, with the shared first-run hint."""

    return _echo_missing(path, what="cache database", hint="it is created by the first run")


def _refused_by_active_run(acquired: bool, data_dir: str) -> bool:
    """True (after echoing why) when another run holds the single-instance lock.

    Touching the cache files while a run is live would clobber the in-flight
    database or snapshot half-committed cycle state, so the cache commands take
    the same lock the runner uses and refuse instead.
    """

    if acquired:
        return False
    hub_error(f"Another Pearlarr run is active in {data_dir} - refusing to modify the cache")
    return True


@contextlib.contextmanager
def _open_cache_readonly(cache_path: str) -> Generator[CacheStore]:
    """Open cache.db read-only for a diagnostic command, closing it afterwards.

    Read-only (no descriptor re-stamp, no WAL switch, no fail-open quarantine) so
    the diagnostic reflects the file as-is. A corrupt/not-a-database file raises
    `sqlite3.DatabaseError` from the first read, for the command to report.
    The caller checks the file exists first (`_echo_missing`).
    """

    from .cache import CacheStore

    store = CacheStore.open_readonly(cache_path)
    try:
        yield store
    finally:
        store.close()


def _trust_os_certificates() -> None:
    """Verify TLS against the OS trust store instead of the bundled certifi CAs.

    `inject_into_ssl` swaps `ssl.SSLContext` for truststore's here at
    the root callback - before any HTTP client builds a context - so a CA
    installed on the host (or handed to a bare container via `SSL_CERT_FILE`)
    is honored by every stack in the process (httpx, requests, urllib).
    """

    import truststore

    truststore.inject_into_ssl()


def _print_version(value: bool) -> None:
    """Eager `--version` callback: print `pearlarr <version>` and exit."""

    if not value:
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        resolved = version("pearlarr")
    except PackageNotFoundError:  # pragma: no cover - only when run from a non-install
        resolved = "unknown"
    typer.echo(f"pearlarr {resolved}")
    raise typer.Exit


# Default command, schedule run
@pearlarr_cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_dir: Annotated[
        str | None,
        typer.Option(
            help="Override the data directory holding config, caches, and logs "
            "(otherwise PEARLARR_DATA_DIR or the OS per-user data directory)",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_print_version,
            is_eager=True,
            help="Print the installed version and exit",
        ),
    ] = False,
) -> None:
    """Pearlarr: sync the anime releases SeaDex recommends into Sonarr and Radarr.

    Without a subcommand, runs in scheduled mode (every configured arr, every
    few hours).
    """

    _trust_os_certificates()

    # The flag is sugar over PEARLARR_DATA_DIR: fold it into the env so every
    # command's resolve_paths() sees it (commands are also called directly in tests,
    # so the env - not ctx.obj - is the single override channel). Flag wins over a
    # pre-set env because it overwrites here, before any subcommand resolves.
    if data_dir is not None:
        os.environ["PEARLARR_DATA_DIR"] = os.path.abspath(data_dir)

    if ctx.invoked_subcommand is None:
        run_scheduled()


_JSON_HELP = "Emit machine-readable JSON events, one object per line"


@pearlarr_cli.command("paths")
def show_paths(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Print the resolved data directory and the files within it."""

    with cli_surface(json_output):
        paths = resolve_paths()
        emit_to_hub(
            PathsShown(
                data_dir=paths.data_dir,
                config=paths.config,
                cache=paths.cache,
                mappings_db=paths.mappings_db,
                log_dir=paths.log_dir,
            ),
        )
        return True


@pearlarr_cli.command("replay")
def replay(
    capture: Annotated[
        str,
        typer.Argument(
            metavar="FILE",
            help="The JSON capture to render (a run's log_format: json output, or a subcommand's --json); "
            "- reads stdin",
        ),
    ],
) -> bool:
    """Re-render a captured JSON event stream as the human text log grammar.

    Reads FILE (or stdin with -), a capture of the machine-readable JSON stream
    written by advanced.log_format: json or a subcommand's --json, and prints
    each event back as one `ts LEVEL [breadcrumb] message key=value` line - the
    same grammar as the log file - for reading a docker-captured log after the
    fact. Non-event lines (docker interleaves stderr text) are skipped with a
    count, and unknown newer events render best-effort. Unlike the other
    subcommands, replay has no --json flag: its output is the rendered text.
    """

    # Lazy like the heavy clients (see the module comment above): replay is off
    # the hot --help / config / cache paths.
    from .replay import replay as run_replay

    with cli_surface(False):
        return run_replay(capture)


_DEFAULT_SCHEDULE_HOURS = 6.0


def _peek_config(config_path: str) -> AppConfig | None:
    """Silently read + validate the config for a pre-logger peek, or None.

    Never writes the starter template and never logs: `bootstrap.load_shared_config`
    owns the first-run copy and the operator-facing config errors. Only the load's
    real failure set (unreadable file, bad YAML, failed validation) is
    suppressed. Anything else is a programming bug and must surface.
    """

    with contextlib.suppress(OSError, yaml.YAMLError, ValidationError):
        if os.path.exists(config_path):
            return AppConfig.load(config_path)
    return None


def _console_format(config_path: str) -> LogFormat:
    """The configured `advanced.log_format` peek, still possibly "auto".

    Folds to "auto" when the config is missing or unreadable - the real load
    owns reporting those. `_resolved_format` folds "auto" to a concrete
    format via `resolve_console_format` (the one fold home).
    """

    peeked = _peek_config(config_path)
    return peeked.advanced.log_format if peeked is not None else "auto"


def _resolved_format(config_path: str) -> LogFormat:
    """The cycle's console format, "auto" folded once (`resolve_console_format`).

    Both run commands call this once per cycle and feed the SAME resolved value
    to `setup_logger` AND `hub.begin_cycle`, so the handler graph and the
    hub's console seat can never disagree within a cycle.
    """

    return resolve_console_format(_console_format(config_path))


def _data_dir_unwritable(data_dir: str, e: OSError) -> NoReturn:
    """Report an unwritable data directory as one actionable stderr line, exit 1.

    These failures strike before any output surface exists (`ensure_data_dir`
    and `_install_output_hub`'s log-dir makedirs run first), so the report
    goes straight to stderr - never a traceback.
    """

    typer.echo(
        f"Cannot write to the data directory {data_dir} ({e}) - fix its permissions, or point "
        f"--data-dir / PEARLARR_DATA_DIR at a writable location",
        err=True,
    )
    raise typer.Exit(1) from None


def _prepare_data_dir(paths: AppPaths) -> None:
    """`ensure_data_dir`, degrading an unwritable location to a clean exit 1."""

    try:
        ensure_data_dir(paths)
    except OSError as e:
        _data_dir_unwritable(paths.data_dir, e)


def _console_seat(console_format: LogFormat, caps_cache: CapsCache) -> Renderer:
    """The hub's console seat for a cycle's RESOLVED format.

    rich gets the cockpit renderer (boot/scan/wait regions over the shared
    Console). Plain and json get the matching stdout text seat.
    """

    # "auto" is unreachable from production (cli resolves pre-begin_cycle).
    # folded defensively for programmatic callers.
    console_format = resolve_console_format(console_format)
    if console_format == "rich":
        return RichRenderer(caps_cache=caps_cache)
    if console_format == "json":
        return JsonRenderer(sys.stdout)
    return LineRenderer(sys.stdout)


def _install_output_hub(paths: AppPaths) -> tuple[OutputHub, FileLogSink]:
    """Build + install the per-process OutputHub. Its sinks own file/plain/json.

    The FileLogSink is deliberately the first stable sink (`_subs[0]`):
    file-before-console dispatch, so a blocked tty can never starve the file.
    Installed BEFORE `setup_logger` - required, so a record fired from inside
    it (the invalid-level complaint) reaches the hub instead of
    `logging.lastResort`. `install_hub` closes any previously installed hub
    (a repeat `run single` in-process must not leak an open FileLogSink).
    The probe is the pre-run writability check: it must reach the log FILE
    itself (a root-owned Pearlarr.log fails open, not makedirs), so an
    unwritable file aborts here at startup instead of silently failing open
    at the sink.
    The FileLogSink is also returned directly (alongside the hub) so a caller
    can drive `apply_retention_days` once the config that names the retention
    window is read - `run_arrs` is the sole reader, config-load being the
    only moment retention is known.
    """

    file_sink = FileLogSink(paths.log_dir)
    try:
        file_sink.probe()
    except OSError as e:
        _data_dir_unwritable(paths.data_dir, e)
    # ONE caps cache shared across console-seat swaps: the seat and its
    # boot/wait regions must branch on the same probe.
    caps_cache = CapsCache()
    hub = OutputHub(
        [file_sink],
        console_factory=partial(_console_seat, caps_cache=caps_cache),
    )
    install_hub(hub)
    install_bridge()
    return hub, file_sink


def _schedule_hours(config_path: str) -> float:
    """Hours between scheduled cycles: SCHEDULE_TIME env (deprecated) > config > 6.

    A valid positive finite SCHEDULE_TIME still wins (with a deprecation
    warning). An invalid one is reported with the value actually used instead.
    Both notices go through the hub so they reach the log file and render
    styled among the run's other lines. Config read failures - including a
    still-missing file - degrade to the default quietly (`_peek_config`).
    """

    raw = os.getenv("SCHEDULE_TIME")
    if raw is not None:
        try:
            hours = float(raw)
        except ValueError:
            hours = math.nan
        if math.isfinite(hours) and hours > 0:
            hub_warn("SCHEDULE_TIME is deprecated - set schedule.interval_hours in the config instead")
            return hours

    peeked = _peek_config(config_path)
    fallback = peeked.schedule.interval_hours if peeked is not None else _DEFAULT_SCHEDULE_HOURS
    if raw is not None:
        hub_warn(f"Invalid SCHEDULE_TIME {raw!r} - using {fallback:g} hours")
    return fallback


def _handle_sigterm(signum: int, frame: FrameType | None) -> NoReturn:
    """Scheduled mode's SIGTERM handler: announce, then exit 0 (a clean stop).

    Docker stop / systemd deliver SIGTERM. The raise interrupts even the
    inter-cycle `time.sleep`, so shutdown is prompt at any point in the loop.
    """

    hub_note("Received SIGTERM - exiting")
    raise SystemExit(0)


@pearlarr_run.command("scheduled")
def run_scheduled(
    log_level: Annotated[
        LogLevel | None,
        typer.Option(case_sensitive=False, help="Override the configured advanced.log_level"),
    ] = None,
) -> None:
    """Run every configured arr module on a loop (each schedule.interval_hours).

    This is the bare-metal fallback scheduler. The container image runs
    supercronic instead.
    """

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    _prepare_data_dir(paths)

    # A stopping container/service must exit promptly and cleanly (code 0), not
    # die mid-sleep with SIGTERM's default nonzero status.
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # The hub is per-process: installed once, BEFORE the loop - install the hub
    # before the logger, see _install_output_hub.
    hub, file_sink = _install_output_hub(paths)

    for cycle in itertools.count(1):
        # The config's console format is re-resolved each cycle (like the
        # cadence below), so a config edit takes effect without a restart.
        console_format = _resolved_format(paths.config)
        logger = setup_logger(log_level=log_level or "INFO", console_format=console_format)
        # The config level lands mid-cycle via apply_log_level -> hub.set_level.
        hub.begin_cycle(console_format=console_format, level=logger.level)
        # Post-begin_cycle, so the boundary lands in the fresh cycle's file.
        emit_to_hub(CycleStarted(number=cycle))

        # Re-read the cadence each cycle so a config edit takes effect without a
        # restart.
        schedule_time = _schedule_hours(paths.config)

        # Build the shared config + id-mapping resolver once and run every
        # configured arr (one config read + one download/parse per cycle, reused
        # by both). On an invalid-config/source failure bootstrap.run_arrs logs
        # the cause and the cycle is skipped, so it's retried next pass rather
        # than crashing. A MISSING config instead exits 1 (typer.Exit from
        # bootstrap.load_shared_config) - retrying can't fill the template in.
        # No ad-hoc preamble here: the branded title (logged by run_arrs) leads
        # each cycle, so the scheduled path reads the same as a single run.
        bootstrap.run_arrs(
            [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)],
            paths=paths,
            logger=logger,
            file_sink=file_sink,
            log_level=log_level,
            retry_note=f"will retry in {schedule_time:g}h (Ctrl-C to stop)",
        )

        # Aware (fixed local offset), so the serialized timestamp carries its
        # UTC offset and matches the sleep-seconds semantics across DST edges.
        emit_to_hub(NextRunScheduled(at=datetime.now().astimezone() + timedelta(hours=schedule_time)))

        time.sleep(schedule_time * 3600)


# Single run. The user-facing help lives on the decorator. The docstring below
# (with its Args block) is for API readers and never reaches --help.
@pearlarr_run.command(
    "single",
    help="Do a single Pearlarr run (every configured arr, unless narrowed by the flags below).",
)
def run_single(
    radarr: Annotated[bool, typer.Option("--radarr", help="Run the Radarr module")] = False,
    sonarr: Annotated[bool, typer.Option("--sonarr", help="Run the Sonarr module")] = False,
    movie_id: Annotated[
        int | None,
        typer.Option(metavar="TMDB_ID", help="Only process the movie with this TMDB ID (implies --radarr)"),
    ] = None,
    series_id: Annotated[
        int | None,
        typer.Option(metavar="TVDB_ID", help="Only process the series with this TVDB ID (implies --sonarr)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Simulate the run: no grabs, no cache writes, no notifications"),
    ] = False,
    import_wait_mode: Annotated[
        ImportWaitMode | None,
        typer.Option(help="Override the configured imports.wait_mode for this run"),
    ] = None,
    log_level: Annotated[
        LogLevel | None,
        typer.Option(case_sensitive=False, help="Override the configured advanced.log_level for this run"),
    ] = None,
) -> bool:
    """Do a single Pearlarr run.

    With no selection flag, every configured arr is run (like scheduled mode).
    The flags narrow the run to one arr or one title.

    Args:
        radarr: Only run the Radarr module.
        sonarr: Only run the Sonarr module.
        movie_id: If set, only run Radarr for the movie with this TMDB ID.
            Implies a Radarr run.
        series_id: If set, only run Sonarr for the series with this TVDB ID.
            Implies a Sonarr run.
        dry_run: If set, simulate the run without grabbing torrents, writing
            the cache, or sending notifications.
        import_wait_mode: Override the configured wait-for-completion + Sonarr
            manual-import mode (off/deferred/blocking/hybrid) for this run. When
            unset the config's `imports.wait_mode` wins (cli > config > default).
        log_level: Override the configured `advanced.log_level` for this run
            (cli > config > INFO).
    """

    # Passing a flag or a movie/series ID narrows the run to that arr. With no
    # selection at all, run everything configured (mirrors scheduled mode). The
    # distinction is remembered so bootstrap.configured_arrs can refuse an explicit
    # request for an unconfigured arr instead of silently skipping it.
    arrs: list[ArrTarget] = []
    if radarr or movie_id is not None:
        arrs.append(ArrTarget(Arr.RADARR, movie_id))
    if sonarr or series_id is not None:
        arrs.append(ArrTarget(Arr.SONARR, series_id))

    explicit_selection = bool(arrs)
    if not arrs:
        arrs = [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)]

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    _prepare_data_dir(paths)

    # Install the hub before the logger - see _install_output_hub.
    console_format = _resolved_format(paths.config)
    hub, file_sink = _install_output_hub(paths)
    logger = setup_logger(log_level=log_level or "INFO", console_format=console_format)
    hub.begin_cycle(console_format=console_format, level=logger.level)

    # Build the shared config + mappings once and run each requested arr. True
    # when the run proceeded and every arr completed. False (exit 1) when the
    # shared config/mappings couldn't be built, the selection was refused, or an
    # arr run failed.
    return bootstrap.run_arrs(
        arrs,
        paths=paths,
        logger=logger,
        file_sink=file_sink,
        explicit_selection=explicit_selection,
        dry_run=dry_run,
        import_wait_mode=import_wait_mode,
        log_level=log_level,
    )


# Config commands
@pearlarr_config.command("init")
def config_init(
    force: Annotated[bool, typer.Option(help="Overwrite an existing config.yml with the starter template")] = False,
    json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False,
) -> bool:
    """Write a starter config.yml to the data directory.

    The file lands in the resolved data directory (see the paths command).
    Override the location with --data-dir or PEARLARR_DATA_DIR. An existing
    config.yml is never overwritten unless --force is passed, so a re-run can't
    wipe out a filled-in configuration.
    """

    # Install the cli surface first so every emit below renders through it. The
    # data-dir failures deliberately stay on the pre-hub stderr path (they also
    # fire pre-hub on the run path, where a hub_error would drop silently).
    with cli_surface(json_output):
        paths = resolve_paths()
        _prepare_data_dir(paths)

        if os.path.exists(paths.config) and not force:
            hub_error(f"{paths.config} already exists - pass --force to overwrite it with the starter template")
            return False

        try:
            write_starter_config(paths.config)
        except OSError as e:
            # A pre-existing data dir gone read-only passes _prepare_data_dir
            # (makedirs is a no-op) and fails here instead.
            _data_dir_unwritable(paths.data_dir, e)
        emit_to_hub(StarterConfigWritten(path=paths.config))

        return True


def _run_config_action[T](path: str, action: Callable[[str], T]) -> T | None:
    """Run a config-file action for a config command, echoing why on failure.

    Unlike the run path (where `AppConfig.load` copies the starter template on
    a missing file), these commands must not create a config, so existence is
    checked first. Every failure mode is a clean echo to stderr, never a
    traceback. The action must fail like `AppConfig.load` does.
    """

    if not os.path.exists(path):
        hub_error(f"No config file at {path} - run pearlarr config init to write a starter template")
        return None
    try:
        return action(path)
    except ValidationError as e:
        hub_error(f"Invalid configuration in {path}:\n{bootstrap.format_validation_errors(e)}")
        return None
    except yaml.YAMLError as e:
        hub_error(f"Unreadable YAML in {path} ({bootstrap.format_yaml_error(e)})")
        return None
    except ConfigRewriteError as e:
        hub_error(str(e))
        return None
    except OSError as e:
        # The error text names the exact file (possibly a .bak/.tmp sibling the
        # migrate action writes), so the advice covers the directory too.
        hub_error(f"Could not access {path} ({e}) - check permissions on it and the data directory")
        return None


@pearlarr_config.command("validate")
def config_validate(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Check config.yml parses and validates, and report what a run would use.

    The status lines call out the settings that silently change a run's shape:
    an unconfigured arr is skipped, and unconfigured qBittorrent credentials
    mean preview mode (nothing is grabbed).
    """

    with cli_surface(json_output):
        paths = resolve_paths()
        app_config = _run_config_action(paths.config, AppConfig.load)
        if app_config is None:
            return False

        outcome = app_config.migration()
        emit_to_hub(
            ConfigValidated(
                path=paths.config,
                migration_notes=outcome.notes if outcome is not None else None,
                sonarr_missing_keys=app_config.missing_arr_keys(Arr.SONARR),
                radarr_missing_keys=app_config.missing_arr_keys(Arr.RADARR),
                qbit_configured=app_config.qbittorrent.credentials() is not None,
            ),
        )
        return True


@pearlarr_config.command("migrate")
def config_migrate(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Rewrite config.yml at the current config schema version, keeping a backup.

    Runs never require this - an older file is migrated in memory at every load -
    but the file itself keeps the old spelling until rewritten. The rewrite is
    the current annotated template with this file's values (and any schema fixes)
    filled in. The previous file is saved beside it as config.yml.bak first. A
    file already at the current version is left untouched.
    """

    with cli_surface(json_output):
        paths = resolve_paths()
        upgrade = _run_config_action(paths.config, upgrade_config_file)
        if upgrade is None:
            return False
        if upgrade.migration is None:
            emit_to_hub(ConfigUpToDate(path=paths.config))
            return True

        # backup_path is set in lockstep with a non-None migration (ConfigUpgrade).
        assert upgrade.backup_path is not None
        emit_to_hub(
            ConfigMigrated(path=paths.config, backup_path=upgrade.backup_path, notes=upgrade.migration.notes),
        )
        return True


# Values under these keys hold credentials (the webhook URLs embed tokens), so
# `config show` masks them. Matched case-insensitively as substrings of the
# dumped key names.
_SECRET_KEY_MARKERS = ("api_key", "password", "webhook", "discord", "username")

# Free-form subtrees that can hide credentials anywhere in their values:
# qbittorrent.options carries arbitrary qbittorrentapi.Client kwargs (proxy
# URLs, auth headers under REQUESTS_ARGS, ...), so every value below one of
# these keys is masked, keeping only the top-level key names.
_MASK_ALL_SUBTREES = ("options",)


def _redact_secrets(node: JsonValue, *, mask_values: bool = False) -> JsonValue:
    """A deep copy of a dumped config with every set secret value masked.

    Only non-None values are masked, so an unset secret still reads as `null`
    (the "is it even set?" question is usually why the dump is being shared).
    URL/host values keep their host but mask any embedded `user:pass@` login.
    `mask_values` (set inside a `_MASK_ALL_SUBTREES` subtree) masks every
    value regardless of key name.
    """

    if isinstance(node, dict):
        redacted: dict[str, JsonValue] = {}
        for key, value in node.items():
            lowered = key.lower()
            if value is not None and (mask_values or any(marker in lowered for marker in _SECRET_KEY_MARKERS)):
                redacted[key] = "REDACTED"
            elif isinstance(value, str) and ("url" in lowered or "host" in lowered):
                redacted[key] = strip_userinfo(value)
            else:
                redacted[key] = _redact_secrets(value, mask_values=mask_values or lowered in _MASK_ALL_SUBTREES)
        return redacted
    if isinstance(node, list):
        return [_redact_secrets(item, mask_values=mask_values) for item in node]
    return node


@pearlarr_config.command("show")
def config_show(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Print the effective config (defaults applied) with secrets redacted.

    Safe to paste into a bug report: values under secret-named keys (api keys,
    passwords, usernames, webhook URLs) are redacted, every value in the
    free-form `qbittorrent.options` block is redacted, a `user:pass@` login
    embedded in a URL/host is redacted, and unset secrets still show as `null`.
    """

    with cli_surface(json_output):
        paths = resolve_paths()
        app_config = _run_config_action(paths.config, AppConfig.load)
        if app_config is None:
            return False

        # Redact before the emit. A model dump is always an object, the assert
        # only narrows the recursive helper's return type.
        redacted = _redact_secrets(app_config.model_dump(mode="json"))
        assert isinstance(redacted, dict)
        emit_to_hub(EffectiveConfigShown(path=paths.config, config=redacted))
        return True


# Cache commands
@pearlarr_cache.command("backup")
def cache_backup(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Back up the cache database to cache.backup.db.

    Uses the SQLite online-backup API so a consistent snapshot is taken even if a
    WAL has uncommitted pages, rather than a raw file copy that could miss them.
    The snapshot lands via temp file + rename, so a failed backup can never
    replace or delete a previous good cache.backup.db.
    """

    with cli_surface(json_output):
        paths = resolve_paths()

        if _echo_missing_cache(paths.cache):
            return False

        with single_instance_lock(paths.data_dir) as acquired:
            if _refused_by_active_run(acquired, paths.data_dir):
                return False

            tmp_backup = paths.cache_backup + ".tmp"
            try:
                with (
                    contextlib.closing(sqlite3.connect(paths.cache)) as source,
                    contextlib.closing(sqlite3.connect(tmp_backup)) as dest,
                ):
                    source.backup(dest)
            except sqlite3.DatabaseError as e:
                # Report the corrupt/torn source cleanly and drop the torn temp file.
                # a previous good cache.backup.db is left untouched.
                with contextlib.suppress(OSError):
                    os.remove(tmp_backup)
                hub_error(f"Cache backup failed ({e}) - run pearlarr cache check to inspect cache.db")
                return False

            try:
                os.replace(tmp_backup, paths.cache_backup)
            except OSError as e:
                hub_error(f"Cache backup failed ({e}) - check the data directory's permissions")
                return False
        emit_to_hub(CacheBackedUp(backup_path=paths.cache_backup))
        return True


@pearlarr_cache.command("restore")
def cache_restore(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Restore the cache database from cache.backup.db, keeping the backup.

    Copy-restore via temp file + atomic swap: the backup survives (so a restore
    is repeatable), cache.db never vanishes mid-restore, and stale WAL/SHM
    sidecars are cleared so the restored snapshot isn't shadowed by them.
    """

    with cli_surface(json_output):
        paths = resolve_paths()

        if _echo_missing(paths.cache_backup, what="backup", hint="run pearlarr cache backup first"):
            return False

        with single_instance_lock(paths.data_dir) as acquired:
            if _refused_by_active_run(acquired, paths.data_dir):
                return False

            tmp_restore = paths.cache + ".tmp"
            try:
                shutil.copyfile(paths.cache_backup, tmp_restore)
                _remove_db_sidecars(paths.cache)
                os.replace(tmp_restore, paths.cache)
            except OSError as e:
                # A read-only data dir / vanished backup: one clean line, no traceback.
                hub_error(f"Cache restore failed ({e}) - cache.db is left unchanged")
                return False

        emit_to_hub(CacheRestored(backup_path=paths.cache_backup))
        return True


@pearlarr_cache.command("remove")
def cache_remove(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Remove the cache database (cache.db and its WAL/SHM sidecars)."""

    with cli_surface(json_output):
        paths = resolve_paths()

        if _echo_missing(paths.cache, what="cache database", hint="nothing to remove"):
            return False

        with single_instance_lock(paths.data_dir) as acquired:
            if _refused_by_active_run(acquired, paths.data_dir):
                return False

            os.remove(paths.cache)
            _remove_db_sidecars(paths.cache)

        emit_to_hub(CacheRemoved(path=paths.cache))
        return True


@pearlarr_cache.command("stats")
def cache_stats(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Print cache health: per-block row counts and on-disk size."""

    with cli_surface(json_output):
        paths = resolve_paths()
        if _echo_missing_cache(paths.cache):
            return False

        with _open_cache_readonly(paths.cache) as store:
            try:
                s = store.stats()
            except sqlite3.DatabaseError as e:
                hub_error(f"Cache stats failed - unreadable database ({e}) - run pearlarr cache check to diagnose it")
                return False

        emit_to_hub(
            CacheStatsReported(
                entries=s.entries,
                torrent_hashes=s.torrent_hashes,
                anilist_meta=s.anilist_meta,
                sonarr_parse=s.sonarr_parse,
                pending_imports=s.pending_imports,
                size_bytes=s.size_bytes,
            ),
        )
        return True


@pearlarr_cache.command("check")
def cache_check(json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False) -> bool:
    """Run a SQLite integrity check on the cache database and print the result."""

    with cli_surface(json_output):
        paths = resolve_paths()
        if _echo_missing_cache(paths.cache):
            return False

        with _open_cache_readonly(paths.cache) as store:
            try:
                result = store.integrity_check()
            except sqlite3.DatabaseError as e:
                # Reporting bad integrity IS this command's job: a result line, not
                # a traceback. To stderr like every failure path (the command exits 1).
                hub_error(f"integrity: {e}")
                return False

        emit_to_hub(CacheIntegrityReported(result=result))
        return True
