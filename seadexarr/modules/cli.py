from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import signal
import sqlite3
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from .boot_view import BootView, make_boot_view
from .config import AppConfig, Arr, LogFormat, config_permissions_loose, restrict_config_permissions, template_path
from .json_narrow import is_json_list, is_json_obj
from .log import LOG_NAME, LogLevel, apply_log_level, indent_string, log_styled, setup_logger
from .manual_import import ImportWaitMode
from .paths import PROJECT_URL, AppPaths, ensure_data_dir, resolve_paths
from .runlock import single_instance_lock

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType

    # Imported only for annotations - the runtime imports live in the functions
    # that use them, so their deps aren't pulled at CLI module load (see below).
    import httpx

    from .cache import CacheStore
    from .mappings import MappingResolver


# The heavy clients (qBittorrent / the SeaDex+httpx chain via cache) are
# imported lazily inside the functions that use them, NOT at module load, so the
# CLI starts and prints its title without paying ~150ms+ for libraries a `--help`
# or a config/cache subcommand never touches. ``ImportWaitMode`` stays eager: it
# rides a typer command signature, which typer resolves at invocation.


def _exit_on_failure(result: object, **_: object) -> None:
    """Turn a command's ``False`` return into exit code 1.

    typer/click ignore return values, so without this every failed command exits
    0 and scripts can't detect the failure. Commands keep returning ``bool`` for
    programmatic callers (tests call them directly, bypassing this callback).
    The ROOT app's registration is the load-bearing one (a sub-command's False
    propagates up through callback-less groups); the sub-app registrations are
    defense-in-depth, so don't "simplify" the root one away.
    """

    if result is False:
        raise typer.Exit(1)


seadexarr_cli = typer.Typer(
    name="seadexarr",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=f"Docs & issues: {PROJECT_URL}",
    result_callback=_exit_on_failure,
)
seadexarr_run = typer.Typer(
    name="run",
    help="Run SeaDexArr: a scheduled loop or a one-off single run.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)
seadexarr_config = typer.Typer(
    name="config",
    help="Initialize, validate or inspect the config file.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)
seadexarr_cache = typer.Typer(
    name="cache",
    help="Back up, restore, remove or inspect the cache database.",
    no_args_is_help=True,
    result_callback=_exit_on_failure,
)

seadexarr_cli.add_typer(seadexarr_run)
seadexarr_cli.add_typer(seadexarr_config)
seadexarr_cli.add_typer(seadexarr_cache)


def _remove_db_sidecars(db_path: str) -> None:
    """Remove a SQLite db's WAL/SHM sidecar files if present (best-effort)."""

    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            os.remove(db_path + suffix)


def _echo_missing(path: str, *, what: str, hint: str) -> bool:
    """True (after echoing why, to stderr) when ``path`` doesn't exist.

    The cache/config commands report a missing file as one line - naming the
    missing ``what`` plus a ``hint`` on how to get one - and a failure exit,
    not a ``FileNotFoundError`` traceback.
    """

    if os.path.exists(path):
        return False
    typer.echo(f"No {what} at {path} - {hint}.", err=True)
    return True


def _echo_missing_cache(path: str) -> bool:
    """``_echo_missing`` for cache.db, with the shared first-run hint."""

    return _echo_missing(path, what="cache database", hint="it is created by the first run")


def _refused_by_active_run(acquired: bool, data_dir: str) -> bool:
    """True (after echoing why) when another run holds the single-instance lock.

    Modifying cache.db while a run is live would clobber the in-flight database,
    so the cache commands take the same lock the runner uses and refuse instead.
    """

    if acquired:
        return False
    typer.echo(
        f"Another SeaDexArr run is active in {data_dir}; refusing to modify the cache.",
        err=True,
    )
    return True


@contextlib.contextmanager
def _open_cache_readonly(cache_path: str) -> Generator[CacheStore]:
    """Open cache.db read-only for a diagnostic command, closing it afterwards.

    Read-only (no descriptor re-stamp, no WAL switch, no fail-open quarantine) so
    the diagnostic reflects the file as-is; a corrupt/not-a-database file raises
    ``sqlite3.DatabaseError`` from the first read, for the command to report.
    The caller checks the file exists first (``_echo_missing``).
    """

    from .cache import CacheStore

    store = CacheStore.open_readonly(cache_path)
    try:
        yield store
    finally:
        store.close()


def _format_validation_errors(e: ValidationError) -> str:
    """The bad keys of a config ValidationError, one indented ``path: message`` line each."""

    return "\n".join(f"  - {'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in e.errors())


def _format_yaml_error(e: yaml.YAMLError) -> str:
    """Describe a YAML parse error from its parts, never via ``str(e)``.

    ``str(e)`` renders a snippet of the offending source line - which IS the
    secret when the syntax error sits on a credential line - so only the
    problem/context text and the line/column position are reported.
    """

    if isinstance(e, yaml.MarkedYAMLError):
        parts = [part for part in (e.context, e.problem) if part]
        description = ", ".join(parts)
        mark = e.problem_mark
        if mark is not None and description:
            description += f" at line {mark.line + 1}, column {mark.column + 1}"
        if description:
            return description
    return type(e).__name__


def _load_shared_config(
    config: str,
    logger: logging.Logger,
    boot: BootView,
    retry: str,
) -> AppConfig | None:
    """Read + validate the config file, once per run (both arrs reuse it).

    Returns None - after logging the specific cause - when the file is invalid
    (the bad keys are listed without a traceback) or unreadable, so the caller
    skips this run and retries next cycle instead of crashing (the user may be
    mid-edit). A MISSING file instead writes the starter template (inside
    ``AppConfig.load``) and exits 1: no retry can succeed until the user fills
    it in, so a scheduled/container run must stop and say so rather than sleep
    on it. ``retry`` is the pre-formatted scheduled-mode note (empty for a
    single run) stating when the loop retries.
    """

    try:
        with boot.step("Reading config"):
            loaded = AppConfig.load(config)
        # An existing config predating the 0600-on-create hardening may still
        # expose its API keys; warn (after the boot step closes) but keep running.
        if config_permissions_loose(config):
            logger.warning(
                f"Config file {config} is readable by other users and holds API keys - "
                f"tighten it with: chmod 600 {config}",
            )
        return loaded
    except FileNotFoundError:
        logger.error(
            f"No config file at {config} - a starter template was written; fill it in and re-run.",
        )
        # typer.Exit is an Exception subclass, but it escapes cleanly: sibling
        # arms can't catch a raise from THIS arm, and _run_arrs wraps this call
        # in try/FINALLY only - so the boot view, web client and run lock all
        # release on the way out to typer (exit code 1).
        raise typer.Exit(1) from None
    except OSError as e:
        # A permissions/FS problem (unreadable file, read-only data dir failing
        # the template copy): report + skip like the invalid-config arms below -
        # exiting would kill a scheduled daemon over a possibly transient error.
        # Must sit after FileNotFoundError (its subclass) and before Exception.
        logger.error(
            f"Could not access config {config} ({e}); check permissions on it and the data directory. "
            f"Skipping this run{retry}.",
        )
    except ValidationError as e:
        # Surface the specific bad keys (nested path -> message) without a traceback,
        # then skip + retry next cycle - same contract as the missing-file branch.
        logger.error(
            f"Invalid configuration in {config}:\n{_format_validation_errors(e)}\n"
            f"Fix the listed keys and re-run. Skipping this run{retry}.",
        )
    except yaml.YAMLError as e:
        # Malformed YAML is a user-facing config problem like a failed validation:
        # a clean report + retry, not the unexpected-error traceback arm below.
        logger.error(
            f"Unreadable YAML in {config} ({_format_yaml_error(e)}). "
            f"Fix the file and re-run. Skipping this run{retry}.",
        )
    except Exception:
        logger.error(f"Could not load config {config}; skipping this run{retry}", exc_info=True)
    return None


def _build_resolver(
    app_config: AppConfig,
    mappings_db: str,
    logger: logging.Logger,
    boot: BootView,
    retry: str,
    web: httpx.Client,
) -> MappingResolver | None:
    """Build the id-mapping resolver both arrs share (settings are arr-independent).

    The resolver downloads-if-stale and (only when a source's content changed)
    parses+indexes the three large mapping sources into ``mappings.db``, then
    serves both arrs from SQL; it is injected (by ``_run_arrs``) into both, so
    that work happens a single time per run and is skipped entirely when the
    sources are unchanged. Returns None - after logging - when a source can't be
    fetched, so the caller skips this run and retries next cycle.
    """

    from .mappings import MappingResolver, MappingSources

    try:
        with boot.step("Refreshing mappings") as mapping_step:
            resolver = MappingResolver(
                cache_time=app_config.advanced.cache_time,
                ignore_anilist_ids=app_config.seadex.ignore_anilist_ids,
                sources=MappingSources(
                    anime=app_config.mappings.anime_mappings,
                    anidb=app_config.mappings.anidb_mappings,
                    anibridge=app_config.mappings.anibridge_mappings,
                ),
                web=web,
                mappings_db=mappings_db,
                logger=logger,
                progress=mapping_step,
            )
            # Overwrite the per-MB download detail with the final "which sources"
            # note, so the graduated line reads e.g. "anime-ids · anidb · anibridge".
            mapping_step.note(resolver.sources_summary())
    except OSError as e:
        # A first-ever source download with no network lands here (a failed
        # refresh of an existing copy falls open inside the resolver): a clean
        # one-liner, not a traceback.
        logger.error(
            f"Could not download the id-mapping sources ({e}); check your network connection. Skipping this run{retry}",
        )
        return None
    except Exception:
        logger.error(
            f"Could not fetch/parse the id-mapping sources; skipping this run{retry}",
            exc_info=True,
        )
        return None

    return resolver


def _configured_arrs(
    arrs: list[tuple[Arr, int | None]],
    app_config: AppConfig,
    *,
    explicit: bool,
    config_path: str,
    logger: logging.Logger,
) -> list[tuple[Arr, int | None]] | None:
    """Drop unconfigured arrs, or refuse when one was explicitly requested.

    A Sonarr-only (or Radarr-only) config is a normal setup: an implicit
    selection (scheduled mode, a flagless ``run single``) skips the unconfigured
    arr with a dim indented note (matching the boot ledger it lands in) instead
    of tripping ``require_connection`` into an "unexpected error" traceback.
    A half-configured arr (url without api_key, or vice versa) is almost
    certainly a mistake, so its skip is a WARNING naming the missing key. An
    explicit ``--radarr``/``--movie-id`` against an unconfigured radarr is a
    config mistake: report it and run nothing. Returns the runnable pairs, or
    None - after logging why - when nothing can run.
    """

    missing = {arr: keys for arr, _ in arrs if (keys := app_config.missing_arr_keys(arr))}
    if explicit and missing:
        for arr, keys in missing.items():
            logger.error(
                f"{arr.capitalize()} was selected but is not configured - set {' and '.join(keys)} in {config_path}",
            )
        return None

    for arr, keys in missing.items():
        if len(keys) == 1:
            other = f"{arr}.api_key" if keys[0] == f"{arr}.url" else f"{arr}.url"
            logger.warning(f"{other} is set but {keys[0]} is not - skipping {arr.capitalize()}")
        else:
            log_styled(logger, indent_string(f"{arr.capitalize()} not configured - skipped"), "grey50")

    kept = [(arr, item_id) for arr, item_id in arrs if arr not in missing]
    if not kept:
        logger.error(f"Neither sonarr nor radarr is configured - set url and api_key for at least one in {config_path}")
        return None
    return kept


def _implicated_arrs(arr: Arr, app_config: AppConfig) -> list[Arr]:
    """The arrs a run leg connects to, for attributing a connection/auth failure.

    A Sonarr leg also builds a Radarr client when ``ignore_movies_in_radarr``
    is on (the specials cross-check), so a connection/auth failure there can
    belong to either instance - the error handlers name every candidate key
    instead of pinning a Radarr outage on Sonarr.
    """

    implicated = [arr]
    if arr is Arr.SONARR and app_config.sonarr.ignore_movies_in_radarr and app_config.is_configured(Arr.RADARR):
        implicated.append(Arr.RADARR)
    return implicated


def _run_arrs(
    arrs: list[tuple[Arr, int | None]],
    *,
    paths: AppPaths,
    logger: logging.Logger,
    explicit_selection: bool = False,
    dry_run: bool = False,
    import_wait_mode: ImportWaitMode | None = None,
    log_level: str | None = None,
    retry_note: str | None = None,
) -> bool:
    """Build the shared config + mappings once, then run each requested arr.

    ``arrs`` is a list of ``(arr_name, item_id)`` pairs; unconfigured arrs are
    dropped (or, when ``explicit_selection`` says the user asked for them by
    flag, refused) via ``_configured_arrs``, and each survivor is run in its own
    try block (which logs and closes independently, so one crashing doesn't ruin
    the other). The shared config read and mapping download/parse happen a single
    time, in that order with the selection check in between, so a run with
    nothing to do fails fast instead of fetching the mapping sources first.
    Returns True when the run proceeded and every arr completed; False - after
    the cause is logged - when the shared deps couldn't be built, nothing
    runnable was selected, or an arr run failed (unreachable/unauthorized arr,
    qBittorrent connection failure, or an unexpected error), so a scripted
    ``run single`` exits non-zero on any failed leg. A MISSING config doesn't
    return at all: ``_load_shared_config`` writes the starter template and
    raises ``typer.Exit(1)`` (see its docstring). An empty ``arrs`` is a
    defensive no-op returning True (both callers guard against it).
    ``import_wait_mode`` is the resolved CLI override threaded into each arr
    (None in scheduled mode); ``log_level`` is the CLI log-level override,
    applied as soon as the config is readable (cli > config > INFO);
    ``retry_note`` is the scheduled-mode retry message (None otherwise).
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
        # Name the resolved data dir in every cycle's log - scheduled mode rotates
        # a fresh log file each cycle, and "which config/cache is this actually
        # using?" is the first support question.
        log_styled(logger, indent_string(f"Data directory: {paths.data_dir}"), "grey50")
        # Pull the heavy run machinery now - after the instant title, before the
        # cockpit's first step - so this one-time import cost lands in the gap
        # between the banner and the spinner rather than stalling a live step.
        from .arr_http import ArrAuthError, ArrConnectionError
        from .cache import CacheSchemaError
        from .run_loop import RunLoop
        from .run_services import QbitConnectionError, RunDeps, RunServices
        from .seadex_radarr import RadarrSync
        from .seadex_sonarr import SonarrSync
        from .seadex_types import BoundaryContractError
        from .web_client import make_web_client

        # One shared client for all non-arr web traffic this cycle (tracker
        # scrapes, AniList, webhooks); both arr legs reuse its pool.
        web = make_web_client()
        try:
            # In scheduled mode retry_note states the loop's next move on every
            # skip (a single run just exits, so there is nothing to append).
            retry = f" - {retry_note}" if retry_note else ""
            app_config = _load_shared_config(paths.config, logger, boot, retry)
            if app_config is None:
                return False
            apply_log_level(logger, log_level or app_config.advanced.log_level)

            # Selection is settled before the mapping fetch, so a refused or
            # empty selection fails fast instead of downloading sources first.
            runnable = _configured_arrs(
                arrs,
                app_config,
                explicit=explicit_selection,
                config_path=paths.config,
                logger=logger,
            )
            if runnable is None:
                return False

            # The parsed/indexed mapping cache lives beside cache.db in the data dir.
            mappings = _build_resolver(app_config, paths.mappings_db, logger, boot, retry, web)
            if mappings is None:
                return False
            all_arrs_completed = True
            try:
                for arr_name, item_id in runnable:
                    # Bound before the try so a RunDeps.build failure can't hit an
                    # UnboundLocalError in the finally's close.
                    deps: RunDeps | None = None
                    try:
                        deps = RunDeps.build(
                            arr_name,
                            paths.cache,
                            logger=logger,
                            mappings=mappings,
                            app_config=app_config,
                            web=web,
                            boot=boot,
                        )
                        services = RunServices(deps, arr_name)
                        runner = RunLoop(deps, services)
                        match arr_name:
                            case Arr.SONARR:
                                runner.run_sync(
                                    SonarrSync(deps, services),
                                    item_id=item_id,
                                    dry_run=dry_run,
                                    import_wait_mode=import_wait_mode,
                                    boot=boot,
                                )
                            case Arr.RADARR:
                                runner.run_sync(
                                    RadarrSync(deps, services),
                                    item_id=item_id,
                                    dry_run=dry_run,
                                    import_wait_mode=import_wait_mode,
                                    boot=boot,
                                )
                    except (QbitConnectionError, CacheSchemaError) as e:
                        # A user-facing environment problem (wrong host/credentials, a
                        # cache.db from a newer release): a clean one-line message, not
                        # a stack trace under "unexpected error". The two arr-client
                        # arms below get the same treatment.
                        all_arrs_completed = False
                        logger.error(str(e))
                    except ArrConnectionError as e:
                        # The error's message names the URL it couldn't reach, which
                        # disambiguates when this leg contacted more than one arr.
                        all_arrs_completed = False
                        keys = " / ".join(f"{a}.url" for a in _implicated_arrs(arr_name, app_config))
                        logger.error(f"{arr_name.capitalize()} run failed: {e} - check {keys} in your config")
                    except BoundaryContractError as e:
                        # The arr answered but its library payload validated to
                        # nothing: a one-line contract error, never a traceback.
                        all_arrs_completed = False
                        logger.error(f"{arr_name.capitalize()} run failed: {e}")
                    except ArrAuthError:
                        all_arrs_completed = False
                        implicated = _implicated_arrs(arr_name, app_config)
                        if len(implicated) == 1:
                            logger.error(
                                f"{arr_name.capitalize()} rejected the API key - check {arr_name}.api_key in your config",
                            )
                        else:
                            # This leg presented more than one key - name every
                            # candidate (the config keys are what the user edits).
                            keys = " / ".join(f"{a}.api_key" for a in implicated)
                            logger.error(
                                f"An arr rejected the API key during the {arr_name.capitalize()} run - "
                                f"check {keys} in your config",
                            )
                    except Exception:
                        all_arrs_completed = False
                        logger.error(f"Unexpected error during {arr_name.capitalize()} run", exc_info=True)
                    finally:
                        # Cap this arr's boot section so the next arr opens a fresh
                        # one; a no-op on the happy path (run_sync already ended it
                        # before scanning), the safety net when a step failed.
                        boot.end_section()
                        if deps is not None:
                            deps.close()
            finally:
                # The resolver owns mappings.db (shared across both arrs); close it once
                # the cycle is done so the connection / WAL handles are released.
                mappings.close()
        finally:
            boot.close()
            web.close()

        return all_arrs_completed


def _trust_os_certificates() -> None:
    """Verify TLS against the OS trust store instead of the bundled certifi CAs.

    ``inject_into_ssl`` swaps :class:`ssl.SSLContext` for truststore's here at
    the root callback - before any HTTP client builds a context - so a CA
    installed on the host (or handed to a bare container via ``SSL_CERT_FILE``)
    is honored by every stack in the process (httpx, requests, urllib).
    """

    import truststore

    truststore.inject_into_ssl()


def _print_version(value: bool) -> None:
    """Eager ``--version`` callback: print ``seadexarr <version>`` and exit."""

    if not value:
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        resolved = version("seadexarr")
    except PackageNotFoundError:  # pragma: no cover - only when run from a non-install
        resolved = "unknown"
    typer.echo(f"seadexarr {resolved}")
    raise typer.Exit


# Default command, schedule run
@seadexarr_cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_dir: Annotated[
        str | None,
        typer.Option(
            help="Override the data directory holding config, caches and logs "
            "(default: SEADEX_ARR_DATA_DIR or the OS per-user data directory).",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_print_version,
            is_eager=True,
            help="Print the installed version and exit.",
        ),
    ] = False,
) -> None:
    """SeaDexArr: sync the best SeaDex-tagged anime releases into Sonarr and Radarr.

    Without a subcommand, runs in scheduled mode (every configured arr, every
    few hours).

    \f
    Args:
        data_dir: Override the data directory holding config, caches and logs
            (typer exposes this as ``--data-dir``). Defaults to None, which uses
            ``SEADEX_ARR_DATA_DIR`` or the OS-standard per-user data location.
        version: Handled entirely by the eager ``_print_version`` callback
            (typer exposes this as ``--version``/``-V``). Defaults to False.
    """

    _trust_os_certificates()

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


_DEFAULT_SCHEDULE_HOURS = 6.0


def _peek_config(config_path: str) -> AppConfig | None:
    """Silently read + validate the config for a pre-logger peek, or None.

    Never writes the starter template and never logs: ``_load_shared_config``
    owns the first-run copy and the user-facing config errors. Only the load's
    real failure set (unreadable file, bad YAML, failed validation) is
    suppressed; anything else is a programming bug and must surface.
    """

    with contextlib.suppress(OSError, yaml.YAMLError, ValidationError):
        if os.path.exists(config_path):
            return AppConfig.load(config_path)
    return None


def _console_format(config_path: str) -> LogFormat:
    """The configured ``advanced.log_format``, for wiring ``setup_logger``.

    Folds to "auto" (resolved by ``setup_logger``) when the config is missing
    or unreadable - the real load owns reporting those.
    """

    peeked = _peek_config(config_path)
    return peeked.advanced.log_format if peeked is not None else "auto"


def _data_dir_unwritable(data_dir: str, e: OSError) -> NoReturn:
    """Report an unwritable data directory as one actionable stderr line, exit 1.

    These failures strike before a logger exists (``ensure_data_dir`` and
    ``setup_logger`` run first), so the report goes straight to stderr - never
    a traceback.
    """

    typer.echo(
        f"Cannot write to the data directory {data_dir} ({e}). "
        f"Fix its permissions, or point --data-dir / SEADEX_ARR_DATA_DIR at a writable location.",
        err=True,
    )
    raise typer.Exit(1) from None


def _prepare_data_dir(paths: AppPaths) -> None:
    """``ensure_data_dir``, degrading an unwritable location to a clean exit 1."""

    try:
        ensure_data_dir(paths)
    except OSError as e:
        _data_dir_unwritable(paths.data_dir, e)


def _setup_run_logger(paths: AppPaths, log_level: LogLevel | None) -> logging.Logger:
    """``setup_logger`` for a run command, with the unwritable-dir treatment.

    The log-dir makedirs and the rotation renames run before any handler exists,
    so an OSError here gets the same clean stderr line as ``ensure_data_dir``.
    """

    try:
        return setup_logger(
            log_level=log_level or "INFO",
            log_dir=paths.log_dir,
            console_format=_console_format(paths.config),
        )
    except OSError as e:
        _data_dir_unwritable(paths.data_dir, e)


def _schedule_hours(config_path: str, logger: logging.Logger) -> float:
    """Hours between scheduled cycles: SCHEDULE_TIME env (deprecated) > config > 6.

    A valid positive finite SCHEDULE_TIME still wins (with a deprecation
    warning); an invalid one is reported with the value actually used instead.
    Both notices go through the logger so they reach the log file and render
    styled among the run's other lines. Config read failures - including a
    still-missing file - degrade to the default quietly (``_peek_config``).
    """

    raw = os.getenv("SCHEDULE_TIME")
    if raw is not None:
        try:
            hours = float(raw)
        except ValueError:
            hours = math.nan
        if math.isfinite(hours) and hours > 0:
            logger.warning("SCHEDULE_TIME is deprecated; set schedule.interval_hours in the config instead.")
            return hours

    peeked = _peek_config(config_path)
    fallback = peeked.schedule.interval_hours if peeked is not None else _DEFAULT_SCHEDULE_HOURS
    if raw is not None:
        logger.warning(f"Invalid SCHEDULE_TIME {raw!r}; using {fallback:g} hours.")
    return fallback


def _handle_sigterm(signum: int, frame: FrameType | None) -> NoReturn:
    """Scheduled mode's SIGTERM handler: log, then exit 0 (a clean stop).

    Docker stop / systemd deliver SIGTERM; the raise interrupts even the
    inter-cycle ``time.sleep``, so shutdown is prompt at any point in the loop.
    """

    logging.getLogger(LOG_NAME).info("Received SIGTERM; exiting.")
    raise SystemExit(0)


@seadexarr_run.command("scheduled")
def run_scheduled(
    log_level: Annotated[
        LogLevel | None,
        typer.Option(case_sensitive=False, help="Override the configured advanced.log_level."),
    ] = None,
) -> None:
    """Run every configured arr module on a loop (each schedule.interval_hours, default 6).

    This is the bare-metal fallback scheduler; containers should use the
    image's built-in scheduler instead.
    """

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    _prepare_data_dir(paths)

    # A stopping container/service must exit promptly and cleanly (code 0), not
    # die mid-sleep with SIGTERM's default nonzero status.
    signal.signal(signal.SIGTERM, _handle_sigterm)

    while True:
        # The config's console format is peeked each cycle (like the cadence
        # below), so a config edit takes effect without a restart.
        logger = _setup_run_logger(paths, log_level)

        # Re-read the cadence each cycle so a config edit takes effect without a
        # restart.
        schedule_time = _schedule_hours(paths.config, logger)

        # Build the shared config + id-mapping resolver once and run every
        # configured arr (one config read + one download/parse per cycle, reused
        # by both). On an invalid-config/source failure _run_arrs logs the cause
        # and the cycle is skipped, so it's retried next pass rather than
        # crashing; a MISSING config instead exits 1 (typer.Exit from
        # _load_shared_config) - retrying can't fill the template in. No ad-hoc
        # preamble here: the branded title (logged by _run_arrs) leads each
        # cycle, so the scheduled path reads the same as a single run.
        _run_arrs(
            [(Arr.RADARR, None), (Arr.SONARR, None)],
            paths=paths,
            logger=logger,
            log_level=log_level,
            retry_note=f"will retry in {schedule_time:g}h (Ctrl-C to stop)",
        )

        # Weekday included: interval_hours can exceed 24, so a bare HH:MM would
        # be ambiguous about which day it means.
        next_run_time = (datetime.now() + timedelta(hours=schedule_time)).strftime("%a %H:%M")
        logger.info(f"Next scheduled run at {next_run_time}")

        time.sleep(schedule_time * 3600)


# Single run. The user-facing help lives on the decorator; the docstring below
# (with its Args block) is for API readers and never reaches --help.
@seadexarr_run.command(
    "single",
    help="Do a single SeaDexArr run (every configured arr, unless narrowed by the flags below).",
)
def run_single(
    radarr: Annotated[bool, typer.Option("--radarr", help="Run the Radarr module.")] = False,
    sonarr: Annotated[bool, typer.Option("--sonarr", help="Run the Sonarr module.")] = False,
    movie_id: Annotated[
        int | None,
        typer.Option(metavar="TMDB_ID", help="Only process the movie with this TMDB ID (implies --radarr)."),
    ] = None,
    series_id: Annotated[
        int | None,
        typer.Option(metavar="TVDB_ID", help="Only process the series with this TVDB ID (implies --sonarr)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Simulate the run: no grabs, no cache writes, no notifications."),
    ] = False,
    import_wait_mode: Annotated[
        ImportWaitMode | None,
        typer.Option(help="Override the configured imports.wait_mode for this run."),
    ] = None,
    log_level: Annotated[
        LogLevel | None,
        typer.Option(case_sensitive=False, help="Override the configured advanced.log_level for this run."),
    ] = None,
) -> bool:
    """Do a single SeaDexArr run.

    With no selection flag, every configured arr is run (like scheduled mode);
    the flags narrow the run to one arr or one title.

    Args:
        radarr: Only run the Radarr module. Defaults to False
        sonarr: Only run the Sonarr module. Defaults to False
        movie_id: If set, only run Radarr for the movie with this TMDB ID.
            Implies a Radarr run. Defaults to None
        series_id: If set, only run Sonarr for the series with this TVDB ID.
            Implies a Sonarr run. Defaults to None
        dry_run: If set, simulate the run without grabbing torrents, writing
            the cache, or sending notifications. Defaults to False
        import_wait_mode: Override the configured wait-for-completion + Sonarr
            manual-import mode (off/deferred/blocking/hybrid) for this run. When
            unset the config's ``imports.wait_mode`` wins (cli > config > default).
        log_level: Override the configured ``advanced.log_level`` for this run
            (cli > config > INFO). Defaults to None (config wins).
    """

    # Passing a flag or a movie/series ID narrows the run to that arr; with no
    # selection at all, run everything configured (mirrors scheduled mode). The
    # distinction is remembered so _configured_arrs can refuse an explicit
    # request for an unconfigured arr instead of silently skipping it.
    arrs: list[tuple[Arr, int | None]] = []
    if radarr or movie_id is not None:
        arrs.append((Arr.RADARR, movie_id))
    if sonarr or series_id is not None:
        arrs.append((Arr.SONARR, series_id))

    explicit_selection = bool(arrs)
    if not arrs:
        arrs = [(Arr.RADARR, None), (Arr.SONARR, None)]

    # Resolve the data directory once and make sure it exists (config-template copy
    # + run lock both need it).
    paths = resolve_paths()
    _prepare_data_dir(paths)

    logger = _setup_run_logger(paths, log_level)

    # Build the shared config + mappings once and run each requested arr. True
    # when the run proceeded and every arr completed; False (exit 1) when the
    # shared config/mappings couldn't be built, the selection was refused, or an
    # arr run failed.
    return _run_arrs(
        arrs,
        paths=paths,
        logger=logger,
        explicit_selection=explicit_selection,
        dry_run=dry_run,
        import_wait_mode=import_wait_mode,
        log_level=log_level,
    )


# Config commands
@seadexarr_config.command("init")
def config_init(
    force: Annotated[bool, typer.Option(help="Overwrite an existing config.yml with the starter template.")] = False,
) -> bool:
    """Write a starter config.yml to the data directory.

    The file lands in the resolved data directory (see the paths command);
    override the location with --data-dir or SEADEX_ARR_DATA_DIR. An existing
    config.yml is never overwritten unless --force is passed, so a re-run can't
    wipe out a filled-in configuration.
    """

    paths = resolve_paths()
    _prepare_data_dir(paths)

    if os.path.exists(paths.config) and not force:
        typer.echo(
            f"{paths.config} already exists; pass --force to overwrite it with the starter template.",
            err=True,
        )
        return False

    try:
        shutil.copyfile(template_path(), paths.config)
        restrict_config_permissions(paths.config)
    except OSError as e:
        # A pre-existing data dir gone read-only passes _prepare_data_dir
        # (makedirs is a no-op) and fails here instead.
        _data_dir_unwritable(paths.data_dir, e)
    typer.echo(f"Wrote a starter config to {paths.config}.")

    return True


def _load_config_reporting(path: str) -> AppConfig | None:
    """Load + validate the config for an inspection command, echoing why on failure.

    Unlike the run path (where ``AppConfig.load`` copies the starter template on
    a missing file), inspecting must not create files, so existence is checked
    first. Every failure mode is a clean echo to stderr, never a traceback.
    """

    if not os.path.exists(path):
        typer.echo(f"No config file at {path}; run `seadexarr config init` to write a starter template.", err=True)
        return None
    try:
        return AppConfig.load(path)
    except ValidationError as e:
        typer.echo(f"Invalid configuration in {path}:\n{_format_validation_errors(e)}", err=True)
        return None
    except yaml.YAMLError as e:
        typer.echo(f"Unreadable YAML in {path}: {_format_yaml_error(e)}", err=True)
        return None
    except OSError as e:
        typer.echo(f"Could not read {path}: {e}", err=True)
        return None


@seadexarr_config.command("validate")
def config_validate() -> bool:
    """Check config.yml parses and validates, and report what a run would use.

    The status lines call out the settings that silently change a run's shape:
    an unconfigured arr is skipped, and unconfigured qBittorrent credentials
    mean preview mode (nothing is grabbed).
    """

    paths = resolve_paths()
    app_config = _load_config_reporting(paths.config)
    if app_config is None:
        return False

    typer.echo(f"OK: {paths.config} is valid.")
    for arr in (Arr.SONARR, Arr.RADARR):
        keys = app_config.missing_arr_keys(arr)
        if not keys:
            status = "configured"
        elif len(keys) == 1:
            # Half-configured is almost certainly a mistake - name the gap here,
            # where the user is actively checking, not just at run time.
            status = f"not configured ({keys[0]} is not set; runs will skip it)"
        else:
            status = "not configured (runs will skip it)"
        typer.echo(f"  {f'{arr}:':<13}{status}")
    qbit_status = (
        "configured" if app_config.qbittorrent.credentials() else "not configured (preview mode: nothing is grabbed)"
    )
    typer.echo(f"  {'qbittorrent:':<13}{qbit_status}")
    return True


# Values under these keys hold credentials (the webhook URLs embed tokens), so
# ``config show`` masks them; matched case-insensitively as substrings of the
# dumped key names.
_SECRET_KEY_MARKERS = ("api_key", "password", "webhook", "discord", "username")

# Free-form subtrees that can hide credentials anywhere in their values:
# qbittorrent.options carries arbitrary qbittorrentapi.Client kwargs (proxy
# URLs, auth headers under REQUESTS_ARGS, ...), so every value below one of
# these keys is masked, keeping only the top-level key names.
_MASK_ALL_SUBTREES = ("options",)


def _strip_userinfo(value: str) -> str:
    """Mask a ``user:pass@`` login embedded in a URL/host config value."""

    scheme, sep, rest = value.partition("://")
    if not sep:
        scheme, rest = "", value
    authority, slash, tail = rest.partition("/")
    if "@" not in authority:
        return value
    prefix = f"{scheme}://" if sep else ""
    return f"{prefix}REDACTED@{authority.rpartition('@')[2]}{slash}{tail}"


def _redact_secrets(node: object, *, mask_values: bool = False) -> object:
    """A deep copy of a dumped config with every set secret value masked.

    Only non-None values are masked, so an unset secret still reads as ``null``
    (the "is it even set?" question is usually why the dump is being shared).
    URL/host values keep their host but mask any embedded ``user:pass@`` login;
    ``mask_values`` (set inside a ``_MASK_ALL_SUBTREES`` subtree) masks every
    value regardless of key name.
    """

    if is_json_obj(node):
        redacted: dict[str, object] = {}
        for key, value in node.items():
            lowered = key.lower()
            if value is not None and (mask_values or any(marker in lowered for marker in _SECRET_KEY_MARKERS)):
                redacted[key] = "REDACTED"
            elif isinstance(value, str) and ("url" in lowered or "host" in lowered):
                redacted[key] = _strip_userinfo(value)
            else:
                redacted[key] = _redact_secrets(value, mask_values=mask_values or lowered in _MASK_ALL_SUBTREES)
        return redacted
    if is_json_list(node):
        return [_redact_secrets(item, mask_values=mask_values) for item in node]
    return node


@seadexarr_config.command("show")
def config_show() -> bool:
    """Print the effective config (defaults applied) with secrets redacted.

    Safe to paste into a bug report: values under secret-named keys (api keys,
    passwords, usernames, webhook URLs) are masked, every value in the
    free-form ``qbittorrent.options`` block is masked, a ``user:pass@`` login
    embedded in a URL/host is masked, and unset secrets still show as ``null``.
    """

    paths = resolve_paths()
    app_config = _load_config_reporting(paths.config)
    if app_config is None:
        return False

    typer.echo(f"# Effective config from {paths.config} (defaults applied, secrets redacted)")
    dump = _redact_secrets(app_config.model_dump(mode="json"))
    typer.echo(yaml.safe_dump(dump, sort_keys=False).rstrip())
    return True


# Cache commands
@seadexarr_cache.command("backup")
def cache_backup() -> bool:
    """Back up the cache database to cache.backup.db.

    Uses the SQLite online-backup API so a consistent snapshot is taken even if a
    WAL has uncommitted pages, rather than a raw file copy that could miss them.
    The snapshot lands via temp file + rename, so a failed backup can never
    replace or delete a previous good cache.backup.db.
    """

    paths = resolve_paths()

    if _echo_missing_cache(paths.cache):
        return False

    tmp_backup = paths.cache_backup + ".tmp"
    source = sqlite3.connect(paths.cache)
    try:
        dest = sqlite3.connect(tmp_backup)
        try:
            source.backup(dest)
        finally:
            dest.close()
    except sqlite3.DatabaseError as e:
        # Report the corrupt/torn source cleanly and drop the torn temp file;
        # a previous good cache.backup.db is left untouched.
        with contextlib.suppress(OSError):
            os.remove(tmp_backup)
        typer.echo(f"cache backup failed: {e}", err=True)
        return False
    finally:
        source.close()

    try:
        os.replace(tmp_backup, paths.cache_backup)
    except OSError as e:
        typer.echo(f"cache backup failed: {e}", err=True)
        return False
    typer.echo(f"Backed up cache to {paths.cache_backup}.")
    return True


@seadexarr_cache.command("restore")
def cache_restore() -> bool:
    """Restore the cache database from cache.backup.db, keeping the backup.

    Copy-restore via temp file + atomic swap: the backup survives (so a restore
    is repeatable), cache.db never vanishes mid-restore, and stale WAL/SHM
    sidecars are cleared so the restored snapshot isn't shadowed by them.
    """

    paths = resolve_paths()

    if _echo_missing(paths.cache_backup, what="backup", hint="run 'seadexarr cache backup' first"):
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
            typer.echo(f"cache restore failed: {e}", err=True)
            return False

    typer.echo(f"Restored cache from {paths.cache_backup}.")
    return True


@seadexarr_cache.command("remove")
def cache_remove() -> bool:
    """Remove the cache database (cache.db and its WAL/SHM sidecars)."""

    paths = resolve_paths()

    if _echo_missing(paths.cache, what="cache database", hint="nothing to remove"):
        return False

    with single_instance_lock(paths.data_dir) as acquired:
        if _refused_by_active_run(acquired, paths.data_dir):
            return False

        os.remove(paths.cache)
        _remove_db_sidecars(paths.cache)

    typer.echo(f"Removed {paths.cache}.")
    return True


@seadexarr_cache.command("stats")
def cache_stats() -> bool:
    """Print cache health: per-block row counts and on-disk size."""

    paths = resolve_paths()
    if _echo_missing_cache(paths.cache):
        return False

    with _open_cache_readonly(paths.cache) as store:
        try:
            s = store.stats()
        except sqlite3.DatabaseError as e:
            typer.echo(f"cache stats: unreadable database ({e})", err=True)
            return False

    rows = [
        ("entries", str(s.entries)),
        ("torrent_hashes", str(s.torrent_hashes)),
        ("anilist_meta", str(s.anilist_meta)),
        ("sonarr_parse", str(s.sonarr_parse)),
        ("pending_imports", str(s.pending_imports)),
        ("size", f"{s.size_bytes / (1024 * 1024):.2f} MiB"),
    ]
    for key, value in rows:
        typer.echo(f"{f'{key}:':<17}{value}")
    return True


@seadexarr_cache.command("check")
def cache_check() -> bool:
    """Run a SQLite integrity check on the cache database and print the result."""

    paths = resolve_paths()
    if _echo_missing_cache(paths.cache):
        return False

    with _open_cache_readonly(paths.cache) as store:
        try:
            result = store.integrity_check()
        except sqlite3.DatabaseError as e:
            # Reporting bad integrity IS this command's job: a result line, not
            # a traceback. To stderr like every failure path (the command exits 1).
            typer.echo(f"integrity: {e}", err=True)
            return False

    typer.echo(f"integrity: {result}")
    return True
