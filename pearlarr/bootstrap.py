"""The composition root: wire a run from config to run loop.

`run_arrs` reads + validates the config once, builds the id-mapping resolver
once (both arrs share them), then wires each requested arr its own
RunDeps -> RunServices -> RunLoop stack and drives it inside an independent
try block, so one arr crashing doesn't ruin the other. `cli.py` (the
presentation layer) calls in; this module never imports it back.

Boot weight: the heavy run machinery stays lazily imported inside the
functions that use it (the boot-cockpit invariant: instant title first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer
import yaml
from pydantic import ValidationError

from .boot_flow import BootFlow
from .config import KNOWN_TRACKERS, AppConfig, Arr, ArrTarget, config_permissions_loose
from .config_migrations import MIGRATE_HINT
from .log import apply_log_level
from .output import FileLogSink, RunFinished, emit_to_hub, hub_error, hub_note, hub_warn
from .runlock import single_instance_lock

if TYPE_CHECKING:
    import logging

    # Imported only for annotations - the runtime imports live in the functions
    # that use them, so their deps aren't pulled at CLI module load.
    import httpx

    from .manual_import import ImportWaitMode
    from .mappings import MappingResolver
    from .paths import AppPaths


def format_validation_errors(e: ValidationError) -> str:
    """The bad keys of a config ValidationError, one indented `path: message` line each."""

    return "\n".join(f"  - {'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in e.errors())


def format_yaml_error(e: yaml.YAMLError) -> str:
    """Describe a YAML parse error from its parts, never via `str(e)`.

    `str(e)` renders a snippet of the offending source line - which IS the
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


def load_shared_config(
    config: str,
    boot: BootFlow,
    retry: str,
) -> AppConfig | None:
    """Read + validate the config file, once per run (both arrs reuse it).

    Returns None - after logging the specific cause - when the file is invalid
    (the bad keys are listed without a traceback) or unreadable, so the caller
    skips this run and retries next cycle instead of crashing (the user may be
    mid-edit). A MISSING file instead writes the starter template (inside
    `AppConfig.load`) and exits 1: no retry can succeed until the user fills
    it in, so a scheduled/container run must stop and say so rather than sleep
    on it. `retry` is the pre-formatted scheduled-mode note (empty for a
    single run) stating when the loop retries.
    """

    try:
        with boot.step("Reading config"):
            loaded = AppConfig.load(config)
        # An existing config predating the 0600-on-create hardening may still
        # expose its API keys; warn (after the boot step closes) but keep running.
        if config_permissions_loose(config):
            hub_warn(
                f"Config file {config} is readable by other users and holds API keys - "
                f"tighten it with: chmod 600 {config}"
            )
        # An unknown tracker name silently matches nothing on SeaDex, so a typo would
        # quietly filter out every release from that tracker - warn, don't reject.
        unknown_trackers = sorted(loaded.seadex.trackers - KNOWN_TRACKERS)
        if unknown_trackers:
            hub_warn(
                f"Unknown seadex.trackers value(s) ignored by matching: "
                f"{', '.join(unknown_trackers)} (known, case-insensitive: "
                f"{', '.join(sorted(KNOWN_TRACKERS))})"
            )
        # An old-schema file loads via the in-memory migration; the warn names
        # what was folded and the command that updates the file itself.
        outcome = loaded.migration()
        if outcome is not None:
            applied = f" ({'; '.join(outcome.notes)})" if outcome.notes else ""
            hub_warn(f"Config file {config} uses an older config schema - migrated in memory{applied} - {MIGRATE_HINT}")
        return loaded
    except FileNotFoundError:
        hub_error(f"No config file at {config} - a starter template was written - fill it in and re-run")
        # typer.Exit is an Exception subclass, but it escapes cleanly: sibling
        # arms can't catch a raise from THIS arm, and run_arrs wraps this call
        # in try/FINALLY only - so the boot view, web client and run lock all
        # release on the way out to typer (exit code 1).
        raise typer.Exit(1) from None
    except OSError as e:
        # A permissions/FS problem (unreadable file, read-only data dir failing
        # the template copy): report + skip like the invalid-config arms below -
        # exiting would kill a scheduled daemon over a possibly transient error.
        # Must sit after FileNotFoundError (its subclass) and before Exception.
        hub_error(
            f"Could not access config {config} ({e}) - check permissions on it and the data directory - "
            f"skipping this run{retry}"
        )
    except ValidationError as e:
        # Surface the specific bad keys (nested path -> message) without a traceback,
        # then skip + retry next cycle - same contract as the missing-file branch.
        hub_error(
            f"Invalid configuration in {config}:\n{format_validation_errors(e)}\n"
            f"Fix the listed keys and re-run - skipping this run{retry}"
        )
    except yaml.YAMLError as e:
        # Malformed YAML is a user-facing config problem like a failed validation:
        # a clean report + retry, not the unexpected-error traceback arm below.
        hub_error(
            f"Unreadable YAML in {config} ({format_yaml_error(e)}) - fix the file and re-run - skipping this run{retry}"
        )
    except Exception as e:
        hub_error(f"Could not load config {config} - skipping this run{retry}", exc=e)
    return None


def build_resolver(
    app_config: AppConfig,
    mappings_db: str,
    logger: logging.Logger,
    boot: BootFlow,
    retry: str,
    web: httpx.Client,
) -> MappingResolver | None:
    """Build the id-mapping resolver both arrs share (settings are arr-independent).

    The resolver downloads-if-stale and (only when a source's content changed)
    parses+indexes the three large mapping sources into `mappings.db`, then
    serves both arrs from SQL; it is injected (by `run_arrs`) into both, so
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
        hub_error(
            f"Could not download the id-mapping sources ({e}) - check your network connection - "
            f"skipping this run{retry}"
        )
        return None
    except Exception as e:
        hub_error(f"Could not fetch/parse the id-mapping sources - skipping this run{retry}", exc=e)
        return None

    return resolver


def configured_arrs(
    arrs: list[ArrTarget],
    app_config: AppConfig,
    *,
    explicit: bool,
    config_path: str,
) -> list[ArrTarget] | None:
    """Drop unconfigured arrs, or refuse when one was explicitly requested.

    A Sonarr-only (or Radarr-only) config is a normal setup: an implicit
    selection (scheduled mode, a flagless `run single`) skips the unconfigured
    arr with a dim note placed in the boot ledger it lands in, instead
    of tripping `require_connection` into an "unexpected error" traceback.
    A half-configured arr (url without api_key, or vice versa) is almost
    certainly a mistake, so its skip is a WARNING naming the missing key. An
    explicit `--radarr`/`--movie-id` against an unconfigured radarr is a
    config mistake: report it and run nothing. Returns the runnable targets, or
    None - after logging why - when nothing can run.
    """

    missing = {target.arr: keys for target in arrs if (keys := app_config.missing_arr_keys(target.arr))}
    if explicit and missing:
        for arr, keys in missing.items():
            hub_error(
                f"{arr.capitalize()} was selected but is not configured - set {' and '.join(keys)} in {config_path}"
            )
        return None

    for arr, keys in missing.items():
        if len(keys) == 1:
            other = f"{arr}.api_key" if keys[0] == f"{arr}.url" else f"{arr}.url"
            hub_warn(f"{other} is set but {keys[0]} is not - skipping {arr.capitalize()}")
        else:
            # Flat message: the rich console indents it via placement (the open
            # boot section); the file/plain surfaces take a structured line.
            hub_note(f"{arr.capitalize()} not configured - skipped")

    kept = [target for target in arrs if target.arr not in missing]
    if not kept:
        hub_error(
            f"Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, or radarr.url and "
            f"radarr.api_key, in {config_path}"
        )
        return None
    return kept


def implicated_arrs(arr: Arr, app_config: AppConfig) -> list[Arr]:
    """The arrs a run leg connects to, for attributing a connection/auth failure.

    A Sonarr leg also builds a Radarr client when `ignore_movies_in_radarr`
    is on (the specials cross-check), so a connection/auth failure there can
    belong to either instance - the error handlers name every candidate key
    instead of pinning a Radarr outage on Sonarr.
    """

    implicated = [arr]
    if arr is Arr.SONARR and app_config.sonarr.ignore_movies_in_radarr and app_config.is_configured(Arr.RADARR):
        implicated.append(Arr.RADARR)
    return implicated


@dataclass(frozen=True)
class LegContext:
    """The run-invariant collaborators every arr leg shares (built once per run).

    Bundles the cross-leg scaffolding so `_run_leg` takes one context plus the
    per-leg `ArrTarget`, instead of a wide positional signature.
    """

    cache: str
    logger: logging.Logger
    mappings: MappingResolver
    app_config: AppConfig
    web: httpx.Client
    boot: BootFlow
    dry_run: bool
    import_wait_mode: ImportWaitMode | None


def _run_leg(target: ArrTarget, ctx: LegContext) -> bool:
    """Run one arr leg end to end: True on a clean completion, False for each caught arm.

    Re-does its own lazy imports of the heavy run machinery (the boot-cockpit
    invariant: nothing heavy at module load). A `BaseException` that matches no
    arm still propagates - the `finally` runs first - so `run_arrs`'s outer
    try/finally chain unwinds exactly as before.
    """

    from .arr_http import ArrAuthError, ArrConnectionError
    from .cache import CacheSchemaError
    from .protocols import ArrSync
    from .run_loop import RunLoop
    from .run_services import QbitConnectionError, RunDeps, RunServices
    from .seadex_radarr import RadarrSync
    from .seadex_sonarr import SonarrSync
    from .seadex_types import ArrItem, BoundaryContractError

    arr_name = target.arr
    item_id = target.item_id
    # Bound before the try so a RunDeps.build failure can't hit an
    # UnboundLocalError in the finally's close.
    deps: RunDeps | None = None
    try:
        # An inner handler so a dying leg's open output frames close
        # BEFORE the except arms below log: a leg-fatal error is a
        # cycle-level fact, not a detail of the entry / item / boot step
        # it died in; a completed leg's single close comes from the run tail.
        try:
            deps = RunDeps.build(
                arr_name,
                ctx.cache,
                logger=ctx.logger,
                mappings=ctx.mappings,
                app_config=ctx.app_config,
                web=ctx.web,
                boot=ctx.boot,
            )
            services = RunServices(deps, arr_name)
            runner = RunLoop(deps, services)

            # The concrete strategy differs per arr, but the run_sync kwargs don't -
            # capture them once so a signature change edits a single place.
            def drive[ItemT: ArrItem](strategy: ArrSync[ItemT]) -> None:
                runner.run_sync(
                    strategy,
                    item_id=item_id,
                    dry_run=ctx.dry_run,
                    import_wait_mode=ctx.import_wait_mode,
                    boot=ctx.boot,
                )

            match arr_name:
                case Arr.SONARR:
                    drive(SonarrSync(deps, services))
                case Arr.RADARR:
                    drive(RadarrSync(deps, services))
        except BaseException:
            # Through the hub seam, not the reporter (deps is None when
            # RunDeps.build itself failed); hub.emit contains renderer errors,
            # so the emit cannot mask the in-flight leg-fatal Exception.
            emit_to_hub(RunFinished(arr=arr_name))
            raise
    except (QbitConnectionError, CacheSchemaError) as e:
        # A user-facing environment problem (wrong host/credentials, a
        # cache.db from a newer release): a clean one-line message, not
        # a stack trace under "unexpected error". The two arr-client
        # arms below get the same treatment.
        hub_error(str(e))
        return False
    except ArrConnectionError as e:
        # The error's message names the URL it couldn't reach, which
        # disambiguates when this leg contacted more than one arr.
        keys = " / ".join(f"{a}.url" for a in implicated_arrs(arr_name, ctx.app_config))
        hub_error(f"{arr_name.capitalize()} run failed - {e} - check {keys} in your config")
        return False
    except BoundaryContractError as e:
        # The arr answered but its library payload validated to
        # nothing: a one-line contract error, never a traceback.
        hub_error(f"{arr_name.capitalize()} run failed - {e}")
        return False
    except ArrAuthError:
        implicated = implicated_arrs(arr_name, ctx.app_config)
        if len(implicated) == 1:
            hub_error(f"{arr_name.capitalize()} rejected the API key - check {arr_name}.api_key in your config")
        else:
            # This leg presented more than one key - name every
            # candidate (the config keys are what the user edits).
            keys = " / ".join(f"{a}.api_key" for a in implicated)
            hub_error(
                f"An arr rejected the API key during the {arr_name.capitalize()} run - check {keys} in your config"
            )
        return False
    except Exception as e:
        hub_error(
            f"Unexpected error during {arr_name.capitalize()} run - skipping the rest of this arr's run",
            exc=e,
        )
        return False
    finally:
        # Cap this arr's boot section so the next arr opens a fresh
        # one; a no-op on the happy path (run_sync already ended it
        # before scanning), the safety net when a step failed.
        ctx.boot.end_section()
        if deps is not None:
            deps.close()
    return True


def run_arrs(
    arrs: list[ArrTarget],
    *,
    paths: AppPaths,
    logger: logging.Logger,
    file_sink: FileLogSink,
    explicit_selection: bool = False,
    dry_run: bool = False,
    import_wait_mode: ImportWaitMode | None = None,
    log_level: str | None = None,
    retry_note: str | None = None,
) -> bool:
    """Build the shared config + mappings once, then run each requested arr.

    Unconfigured arrs are dropped (or, when `explicit_selection` says the user
    asked for them by flag, refused) via `configured_arrs`, and each survivor is
    run in its own `_run_leg` (which logs and closes independently, so one
    crashing doesn't ruin the other). The shared config read and mapping
    download/parse happen a single time, in that order with the selection check
    in between, so a run with nothing to do fails fast instead of fetching the
    mapping sources first.
    Returns True when the run proceeded and every arr completed; False - after
    the cause is logged - when the shared deps couldn't be built, nothing
    runnable was selected, or an arr run failed (unreachable/unauthorized arr,
    qBittorrent connection failure, or an unexpected error), so a scripted
    `run single` exits non-zero on any failed leg. A MISSING config doesn't
    return at all: `load_shared_config` writes the starter template and
    raises `typer.Exit(1)` (see its docstring). An empty `arrs` is a
    defensive no-op returning True (both callers guard against it).
    `file_sink` is the installed hub's FileLogSink (`cli._install_output_hub`
    returns it alongside the hub) - its retention window is only knowable once
    `app_config` is read, so it rides in as its own dep rather than through the
    hub's global registry. `import_wait_mode` is the resolved CLI override
    threaded into each arr (None in scheduled mode); `log_level` is the CLI
    log-level override, applied as soon as the config is readable
    (cli > config > INFO); `retry_note` is the scheduled-mode retry message
    (None otherwise).
    """

    if not arrs:
        return True

    # Guard against two runs sharing one data directory (cache.db + WAL); SQLite
    # keeps the file safe, but overlapping runs would duplicate work and could race
    # on imports. A different data dir gets its own lock, so intentional parallel
    # instances are still fine.
    with single_instance_lock(paths.data_dir, logger=logger) as acquired:
        if not acquired:
            hub_warn(f"Another Pearlarr run is active in {paths.data_dir} - skipping this run")
            return False

        # The banner names the data dir every cycle: scheduled mode rotates the
        # log per cycle, and "which data dir is this?" is the first support question.
        boot = BootFlow(paths.data_dir)
        boot.banner()
        # The heavy run machinery is imported inside `_run_leg` (the boot-cockpit
        # invariant: nothing heavy at module load); only the shared web client is
        # needed out here.
        from .web_client import make_web_client

        # One shared client for all non-arr web traffic this cycle (tracker
        # scrapes, AniList, webhooks); both arr legs reuse its pool.
        web = make_web_client()
        try:
            # In scheduled mode retry_note states the loop's next move on every
            # skip (a single run just exits, so there is nothing to append).
            retry = f" - {retry_note}" if retry_note else ""
            app_config = load_shared_config(paths.config, boot, retry)
            if app_config is None:
                return False
            apply_log_level(logger, log_level or app_config.advanced.log_level)
            file_sink.apply_retention_days(app_config.advanced.log_retention_days)

            # Selection is settled before the mapping fetch, so a refused or
            # empty selection fails fast instead of downloading sources first.
            runnable = configured_arrs(
                arrs,
                app_config,
                explicit=explicit_selection,
                config_path=paths.config,
            )
            if runnable is None:
                return False

            # The parsed/indexed mapping cache lives beside cache.db in the data dir.
            mappings = build_resolver(app_config, paths.mappings_db, logger, boot, retry, web)
            if mappings is None:
                return False
            all_arrs_completed = True
            ctx = LegContext(
                cache=paths.cache,
                logger=logger,
                mappings=mappings,
                app_config=app_config,
                web=web,
                boot=boot,
                dry_run=dry_run,
                import_wait_mode=import_wait_mode,
            )
            try:
                for target in runnable:
                    # Each leg logs + closes independently (a False return = a caught
                    # arm), so one crashing doesn't drop the other.
                    if not _run_leg(target, ctx):
                        all_arrs_completed = False
            finally:
                # The resolver owns mappings.db (shared across both arrs); close it once
                # the cycle is done so the connection / WAL handles are released.
                mappings.close()
        finally:
            boot.close()
            web.close()

        return all_arrs_completed
