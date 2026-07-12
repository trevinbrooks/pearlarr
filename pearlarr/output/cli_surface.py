"""The subcommand output surface: a hub seat + install shim for the typer commands.

The `config`/`cache`/`paths` subcommands emit the same typed events run commands
do, through a minimal hub carrying ONE console seat and no file sink (subcommands
must work with an unwritable log dir). `cli_surface` installs that hub for a
command body; `--json` picks the JSON seat, otherwise `CliTextRenderer`
reproduces today's `typer.echo` output byte-for-byte (so the docs and tests that
pin it verbatim keep passing). The seat is built INSIDE the contextmanager so
`sys.stdout` resolves after any pytest/CliRunner stream swap.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, ClassVar, assert_never, final

import typer
import yaml

from .events import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CacheBackedUp,
    CacheIntegrityReported,
    CacheRemoved,
    CacheRestored,
    CacheStatsReported,
    CapReached,
    ConfigMigrated,
    ConfigUpToDate,
    ConfigValidated,
    CycleStarted,
    Diagnostic,
    EffectiveConfigShown,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    LedgerRow,
    NextRunScheduled,
    PathsShown,
    ReleaseSkipped,
    RunFinished,
    RunStarted,
    RunSummaryReady,
    ScanFinished,
    ScanStarted,
    ScopeClosed,
    ScopeOpened,
    Severity,
    StarterConfigWritten,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)
from .hub import OutputHub
from .runtime import install_hub, uninstall_hub
from .textline import JsonRenderer
from ..config import Arr
from ..config_migrations import MIGRATE_HINT

if TYPE_CHECKING:
    from collections.abc import Generator


@final
class CliTextRenderer:
    """The subcommands' human seat: reproduces today's `typer.echo` output verbatim.

    A WARNING+ diagnostic goes to stderr (so `config show > cfg.yml` stays
    clean); everything else prints to stdout via `typer.echo` (keeping click's
    Windows-console encoding behavior identical). Run-lifecycle events can never
    reach this seat, so they render nothing.
    """

    writes_file_only: ClassVar[bool] = False

    def handle(self, event: Event, when: float) -> None:
        del when
        match event:
            case PathsShown():
                self._paths(event)
            case StarterConfigWritten(path=path):
                typer.echo(f"Wrote a starter config to {path}")
            case ConfigValidated():
                self._config_validated(event)
            case ConfigUpToDate(path=path):
                typer.echo(f"{path} is already at the current config schema - nothing to do")
            case ConfigMigrated():
                self._config_migrated(event)
            case EffectiveConfigShown(path=path, config=config):
                typer.echo(f"# Effective config from {path} (defaults applied, secrets redacted)")
                typer.echo(yaml.safe_dump(config, sort_keys=False).rstrip())
            case CacheBackedUp(backup_path=backup_path):
                typer.echo(f"Backed up cache to {backup_path}")
            case CacheRestored(backup_path=backup_path):
                typer.echo(f"Restored cache from {backup_path}")
            case CacheRemoved(path=path):
                typer.echo(f"Removed {path}")
            case CacheStatsReported():
                self._cache_stats(event)
            case CacheIntegrityReported(result=result):
                typer.echo(f"integrity: {result}")
            case Diagnostic(severity=severity, message=message):
                typer.echo(message, err=severity >= Severity.WARNING)
            case (
                RunStarted()
                | CycleStarted()
                | NextRunScheduled()
                | ScopeOpened()
                | ScopeClosed()
                | BootStepStarted()
                | BootStepProgressed()
                | BootStepSlow()
                | BootStepFinished()
                | BootReady()
                | ScanStarted()
                | ItemStarted()
                | EntryHeader()
                | EntryDetail()
                | LedgerRow()
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | ScanFinished()
                | RunSummaryReady()
                | WaitStarted()
                | WaitProgress()
                | TorrentGraduated()
                | WaitFinished()
                | RunFinished()
            ):
                # Run-lifecycle events never reach the cli seat.
                pass
            case _:
                assert_never(event)

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def _paths(event: PathsShown) -> None:
        typer.echo(f"data_dir:    {event.data_dir}")
        typer.echo(f"config:      {event.config}")
        typer.echo(f"cache:       {event.cache}")
        typer.echo(f"mappings_db: {event.mappings_db}")
        typer.echo(f"logs:        {event.log_dir}")

    @staticmethod
    def _config_validated(event: ConfigValidated) -> None:
        typer.echo(f"OK: {event.path} is valid")
        if event.migration_notes is not None:
            typer.echo(f"  {'schema:':<13}older config schema, migrated in memory at load - {MIGRATE_HINT}")
            for note in event.migration_notes:
                typer.echo(f"  {'':<13}- {note}")
        for arr, keys in ((Arr.SONARR, event.sonarr_missing_keys), (Arr.RADARR, event.radarr_missing_keys)):
            if not keys:
                status = "configured"
            elif len(keys) == 1:
                status = f"not configured ({keys[0]} is not set - runs will skip it)"
            else:
                status = "not configured (runs will skip it)"
            typer.echo(f"  {f'{arr}:':<13}{status}")
        qbit_status = "configured" if event.qbit_configured else "not configured (preview mode - nothing is grabbed)"
        typer.echo(f"  {'qbittorrent:':<13}{qbit_status}")

    @staticmethod
    def _config_migrated(event: ConfigMigrated) -> None:
        typer.echo(f"Migrated {event.path} to the current config schema - previous file saved as {event.backup_path}")
        for note in event.notes:
            typer.echo(f"  - {note}")

    @staticmethod
    def _cache_stats(event: CacheStatsReported) -> None:
        rows = [
            ("entries", str(event.entries)),
            ("torrent_hashes", str(event.torrent_hashes)),
            ("anilist_meta", str(event.anilist_meta)),
            ("sonarr_parse", str(event.sonarr_parse)),
            ("pending_imports", str(event.pending_imports)),
            ("size", f"{event.size_bytes / (1024 * 1024):.2f} MiB"),
        ]
        for key, value in rows:
            typer.echo(f"{f'{key}:':<17}{value}")


@contextlib.contextmanager
def cli_surface(json_output: bool) -> Generator[None]:
    """Install a minimal one-seat hub for a subcommand body, closing it on exit.

    No file sink, no bridge, no begin_cycle: INFO renders straight through the
    seat with the hub's synchronous same-thread drain. Closing the hub on exit is
    the output barrier before the command's exit code fires.
    """

    seat = JsonRenderer(sys.stdout) if json_output else CliTextRenderer()
    hub = OutputHub([], console=seat)
    install_hub(hub)
    try:
        yield
    finally:
        uninstall_hub()
