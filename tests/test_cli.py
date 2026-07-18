# pyright: strict
# pyright: reportPrivateUsage=false
# `_schedule_hours` is a deliberately under-test private helper. The repo already
# disables reportPrivateUsage for all of tests/, but the strict directive above
# re-enables it, so restore the repo's test policy here.
"""Tests for the Pearlarr CLI commands.

Pins the behaviors the commands must guarantee:

* A corrupt / not-a-database `cache.db` is *reported*, never crashed on - the
  `stats` and `check` diagnostics (and `backup`) return a clean False instead
  of letting a `sqlite3` traceback escape (finding #4). `check` exists to report
  bad integrity, so it must survive the very corruption it diagnoses.
* The destructive commands (`restore` / `remove`) take the single-instance run
  lock first and refuse while a run is active, so they never unlink or replace the
  live db out from under it (finding #5).
* Failure paths report cleanly (missing files echo one hint line, not a traceback),
  failure text goes to stderr (so `pearlarr config show > cfg.yml` stays clean)
  while success output stays on stdout, and a False return maps to exit code 1
  through the Typer apps' result callback.
* `config init` never overwrites a filled-in config.yml without `--force`.
* `run single` with no selection flag runs every *configured* arr (scheduled-mode
  symmetry). An explicit flag for an unconfigured arr refuses cleanly, an implicit
  selection skips it with a dim ledger note - except a half-configured arr (url
  without api_key), whose skip warns by name (`configured_arrs`).
* The inspection commands (`config validate` / `config show`) never write the
  starter template, and `show` masks secret-named values (plus the free-form
  `qbittorrent.options` block and URL-embedded logins) while keeping unset
  ones `null`.
* `advanced.log_level` is applied to CLI runs as soon as the config is read,
  and `--log-level` overrides it (cli > config).
* A missing config exits 1 with the starter template written (no silent
  skip-and-retry). An invalid config keeps the skip+retry contract in scheduled
  mode. SIGTERM stops the scheduled loop with exit code 0.

Each test points `resolve_paths()` at its own `tmp_path` via `PEARLARR_DATA_DIR`
and calls the command functions directly (they return `bool`). The exit-code
tests go through `CliRunner` since the callback only runs inside typer.
"""

import io
import json
import logging
import os
import re
import signal
import ssl
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import ClassVar, NoReturn, cast, override

import pytest
import truststore
from typer.testing import CliRunner

from pearlarr.boot_flow import BootFlow
from pearlarr.bootstrap import configured_arrs, load_shared_config, run_arrs
from pearlarr.cache import CacheStore
from pearlarr.cli import (
    _console_format,
    _console_seat,
    _handle_sigterm,
    _resolved_format,
    _schedule_hours,
    _trust_os_certificates,
    cache_backup,
    cache_check,
    cache_remove,
    cache_restore,
    cache_stats,
    config_init,
    config_migrate,
    config_show,
    config_validate,
    pearlarr_cli,
    run_scheduled,
    run_single,
)
from pearlarr.config import AppConfig, Arr, ArrTarget, LogFormat, template_path
from pearlarr.config_migrations import CONFIG_VERSION
from pearlarr.console_caps import CapsCache
from pearlarr.log import (
    LOG_NAME,
    HubBridgeBase,
    LogLevel,
    RichConsoleHandler,
    apply_log_level,
    setup_logger,
)
from pearlarr.manual_import import ImportWaitMode, Outcome
from pearlarr.output import (
    CycleStarted,
    Diagnostic,
    Event,
    FileLogSink,
    ItemStarted,
    NextRunScheduled,
    OutputHub,
    Renderer,
    RunStarted,
    ScanStarted,
    Severity,
    TorrentGraduated,
    install_hub,
)
from pearlarr.output.recording import RecordingHub
from pearlarr.paths import AppPaths, resolve_paths
from pearlarr.runlock import single_instance_lock

from .builders import make_logger
from .fakes import TtyStringIO, diagnostic_messages, install_recording_hub
from .test_scan_parity import SUMMARY_MINIMAL


def _starter_text() -> str:
    """What a user's starter copy holds: the template minus its generated-file banner."""

    template = Path(template_path()).read_text(encoding="utf-8")
    return "".join(line for line in template.splitlines(keepends=True) if not line.startswith("# GENERATED"))


def _build_cache(tmp_path: Path) -> None:
    """Write a real on-disk `cache.db` under `tmp_path` holding one entry.

    Uses the normal load -> stage -> `save(preview=False)` path, which promotes
    the in-memory db to the file, so the fixture matches a cache a real run leaves
    behind (entry `(SONARR, 7)` with `name="X"`).
    """

    store = CacheStore.load(str(tmp_path / "cache.db"), config_checksum="x")
    store.update_cache(Arr.SONARR, 7, {"name": "X"})
    store.save(preview=False)
    store.close()


class TestCacheRoundTrip:
    """`cache_backup`/`cache_restore`/`cache_remove` round-trip and clean up the cache db.

    Backup-then-restore preserves data and clears a stale WAL/SHM sidecar. Remove deletes the db and its sidecars.
    """

    def test_backup_then_restore_preserves_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_backup() is True
        assert (tmp_path / "cache.backup.db").exists()
        # Success is confirmed out loud (on stdout), not silently.
        assert f"Backed up cache to {tmp_path / 'cache.backup.db'}" in capsys.readouterr().out

        # A stale WAL left next to cache.db must not shadow the restored snapshot.
        # restore clears the sidecars before swapping the copy into place.
        (tmp_path / "cache.db-wal").write_text("stale")

        assert cache_restore() is True
        assert f"Restored cache from {tmp_path / 'cache.backup.db'}" in capsys.readouterr().out
        # Copy-restore: the backup SURVIVES (a post-restore corruption can be
        # restored again) and no temp file is left behind.
        assert (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.db.tmp").exists()
        assert not (tmp_path / "cache.db-wal").exists()  # stale sidecar cleared

        # Restore is repeatable off the surviving backup.
        assert cache_restore() is True
        capsys.readouterr()  # drain the second success echo (keeps it off the terminal under -s)

        # The restored db still holds the original entry.
        store = CacheStore.open_readonly(str(tmp_path / "cache.db"))
        try:
            entry = store.get_entry(Arr.SONARR, 7)
            assert entry is not None
            assert entry.name == "X"
        finally:
            store.close()

    def test_remove_deletes_db_and_sidecars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        (tmp_path / "cache.db-wal").write_text("stale")
        (tmp_path / "cache.db-shm").write_text("stale")

        assert cache_remove() is True
        assert f"Removed {tmp_path / 'cache.db'}" in capsys.readouterr().out
        assert not (tmp_path / "cache.db").exists()
        assert not (tmp_path / "cache.db-wal").exists()
        assert not (tmp_path / "cache.db-shm").exists()


class TestHealthyDiagnostics:
    """`cache_stats` and `cache_check` both succeed on a healthy db and report its integrity as ok."""

    def test_stats_and_check_report_a_healthy_db(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_stats() is True
        assert cache_check() is True
        assert "integrity: ok" in capsys.readouterr().out


class TestCorruptDatabaseIsReportedNotCrashed:
    """Finding #4: a corrupt cache.db is reported cleanly, never tracebacks."""

    def test_check_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # Reporting bad integrity is this command's whole job, so it must not crash
        # on the very corruption it diagnoses. The failure line goes to stderr.
        assert cache_check() is False
        assert "integrity" in capsys.readouterr().err

    def test_stats_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        assert cache_stats() is False
        assert "Cache stats failed" in capsys.readouterr().err

    def test_backup_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # backup reads the source through the online-backup API, so a corrupt source
        # surfaces as a clean failure line (on stderr), not a traceback.
        assert cache_backup() is False
        assert "Cache backup failed" in capsys.readouterr().err


class TestActiveRunGuard:
    """Finding #5: cache commands refuse while a run holds the lock."""

    def test_remove_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        # Holding the single-instance lock on the data dir simulates an active run.
        # remove must refuse and leave the live db untouched.
        with single_instance_lock(str(tmp_path)):
            assert cache_remove() is False
        assert (tmp_path / "cache.db").exists()
        # The refusal is a user-facing stderr line, not a silent no-op (capturing
        # it also keeps it off the terminal under `-s`).
        assert "another pearlarr run is active" in capsys.readouterr().err.lower()

    def test_restore_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        with single_instance_lock(str(tmp_path)):
            assert cache_restore() is False
        # Refused before touching anything: the backup is still there to restore.
        assert (tmp_path / "cache.backup.db").exists()
        assert (tmp_path / "cache.db").exists()
        # The refusal is a user-facing stderr line, not a silent no-op (capturing
        # it also keeps it off the terminal under `-s`).
        assert "another pearlarr run is active" in capsys.readouterr().err.lower()

    def test_backup_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        # A mid-run snapshot would capture half-committed cycle state. Backup
        # refuses like restore/remove and writes nothing.
        with single_instance_lock(str(tmp_path)):
            assert cache_backup() is False
        assert not (tmp_path / "cache.backup.db").exists()
        assert "another pearlarr run is active" in capsys.readouterr().err.lower()


class TestMissingFilesAreReportedNotRaised:
    """A missing cache/backup file echoes one hinting stderr line and returns False."""

    def test_each_cache_command_reports_a_missing_file_with_a_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))

        # Each message names what's missing and hints at how to get one, so a
        # fresh install isn't met with a bare "No file at ...".
        checks: list[tuple[Callable[[], bool], str]] = [
            (cache_backup, "No cache database at"),
            (cache_stats, "No cache database at"),
            (cache_check, "No cache database at"),
            (cache_backup, "it is created by the first run"),
            (cache_remove, "nothing to remove"),
            (cache_restore, "No backup at"),
            (cache_restore, "run pearlarr cache backup first"),
        ]
        for command, expected in checks:
            assert command() is False
            captured = capsys.readouterr()
            # One stderr line, nothing on stdout (scripted use stays clean).
            assert expected in captured.err
            assert captured.err.count("\n") == 1
            assert captured.out == ""

    def test_failed_backup_leaves_no_partial_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # A torn snapshot must not survive a failed backup: a later restore would
        # move it over the live database.
        assert cache_backup() is False
        assert "Cache backup failed" in capsys.readouterr().err
        assert not (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.backup.db.tmp").exists()

    def test_failed_backup_preserves_the_previous_good_backup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        # The live db going corrupt is the very scenario backups exist for. The
        # failed re-backup must leave the good snapshot restorable.
        (tmp_path / "cache.db").write_text("not a database")
        assert cache_backup() is False
        # The failure line itself is pinned by test_failed_backup_leaves_no_partial_snapshot.
        # here we only drain it so the real echo stays off the terminal under `-s`.
        capsys.readouterr()

        store = CacheStore.open_readonly(str(tmp_path / "cache.backup.db"))
        try:
            entry = store.get_entry(Arr.SONARR, 7)
            assert entry is not None
            assert entry.name == "X"
        finally:
            store.close()


class TestConfigInit:
    """config init writes the starter template but never clobbers without --force."""

    def test_init_writes_then_refuses_then_forces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        config = tmp_path / "config.yml"

        assert config_init() is True
        assert config.exists()
        if os.name == "posix":
            # The starter will be filled with API keys: it must land owner-only.
            assert (config.stat().st_mode & 0o777) == 0o600

        # A filled-in config must survive an accidental re-run. The refusal is
        # a stderr line.
        config.write_text("sonarr: {url: http://mine}")
        assert config_init() is False
        assert config.read_text() == "sonarr: {url: http://mine}"
        assert "--force" in capsys.readouterr().err

        assert config_init(force=True) is True
        assert config.read_text() == _starter_text()
        # The write is proven by the file content above. Drain the post-force echo so
        # it can't spill to the terminal under `-s` (output after a mid-test
        # readouterr() is flushed there at teardown when global capture is off).
        capsys.readouterr()


class _RunArrsRecorder:
    """A stand-in for `bootstrap.run_arrs` that records how `run_single` calls it."""

    def __init__(self) -> None:
        self.arrs: list[tuple[Arr, int | None]] | None = None
        self.explicit_selection: bool | None = None
        self.log_level: str | None = None

    def __call__(
        self,
        arrs: list[tuple[Arr, int | None]],
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
        self.arrs = arrs
        self.explicit_selection = explicit_selection
        self.log_level = log_level
        return True


class TestRunSingleSelection:
    """No selection flag = every arr, implicitly. Any flag/id narrows explicitly.

    `run_arrs` is faked out, so these pin only the selection wiring: which
    `(arr, item_id)` pairs are requested and whether the request counts as
    explicit (an explicit request for an unconfigured arr must refuse rather
    than skip - that arm is pinned on `configured_arrs` below).
    """

    @pytest.fixture
    def recorder(self, monkeypatch: pytest.MonkeyPatch) -> _RunArrsRecorder:
        recorder = _RunArrsRecorder()
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", recorder)
        return recorder

    def test_no_selection_requests_both_arrs_implicitly(self, recorder: _RunArrsRecorder) -> None:
        assert run_single() is True
        assert recorder.arrs == [(Arr.RADARR, None), (Arr.SONARR, None)]
        assert recorder.explicit_selection is False

    def test_a_module_flag_narrows_and_is_explicit(self, recorder: _RunArrsRecorder) -> None:
        assert run_single(sonarr=True) is True
        assert recorder.arrs == [(Arr.SONARR, None)]
        assert recorder.explicit_selection is True

    def test_an_id_implies_its_arr(self, recorder: _RunArrsRecorder) -> None:
        assert run_single(movie_id=42) is True
        assert recorder.arrs == [(Arr.RADARR, 42)]
        assert recorder.explicit_selection is True

    def test_the_log_level_flag_is_threaded_through(self, recorder: _RunArrsRecorder) -> None:
        assert run_single(log_level=LogLevel.DEBUG) is True
        assert recorder.log_level == LogLevel.DEBUG


def _config_with(*, sonarr: bool = False, radarr: bool = False) -> AppConfig:
    """An in-memory config with the requested arrs' connection pairs filled in."""

    data: dict[str, object] = {}
    if sonarr:
        data["sonarr"] = {"url": "http://sonarr:8989", "api_key": "k"}
    if radarr:
        data["radarr"] = {"url": "http://radarr:7878", "api_key": "k"}
    return AppConfig.model_validate(data)


class TestConfiguredArrs:
    """Unconfigured arrs: implicit selections skip them, explicit ones refuse."""

    @pytest.fixture
    def recording(self) -> RecordingHub:
        return install_recording_hub()

    def test_implicit_selection_skips_the_unconfigured_arr(self, recording: RecordingHub) -> None:
        kept = configured_arrs(
            [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)],
            _config_with(sonarr=True),
            explicit=False,
            config_path="config.yml",
        )
        assert kept == [(Arr.SONARR, None)]
        # The note is a first-party INFO Diagnostic on the hub: a Sonarr-only
        # setup is normal, not an error. The message is FLAT (the rich console
        # indents it via placement inside the open boot section). The arr name
        # is capitalized prose, not a lowercase config key.
        (skip,) = recording.of_type(Diagnostic)
        assert skip.severity is Severity.INFO
        assert skip.message == "Radarr not configured - skipped"
        assert skip.origin == LOG_NAME
        assert not skip.file_only

    def test_a_half_configured_arr_warns_by_name(self, recording: RecordingHub) -> None:
        config = AppConfig.model_validate(
            {"sonarr": {"url": "http://sonarr:8989", "api_key": "k"}, "radarr": {"url": "http://radarr:7878"}},
        )
        kept = configured_arrs(
            [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)],
            config,
            explicit=False,
            config_path="config.yml",
        )
        assert kept == [(Arr.SONARR, None)]
        # url XOR api_key is almost certainly an accident: the skip must be loud
        # and name the missing half, not read like an intentional single-arr setup.
        # Dotted keys stay lowercase. The prose subject is capitalized.
        (warning,) = recording.of_type(Diagnostic)
        assert warning.severity is Severity.WARNING
        assert warning.message == "radarr.url is set but radarr.api_key is not - skipping Radarr"

    def test_explicit_selection_of_a_half_configured_arr_names_only_the_missing_key(
        self,
        recording: RecordingHub,
    ) -> None:
        config = AppConfig.model_validate({"radarr": {"url": "http://radarr:7878"}})
        kept = configured_arrs([ArrTarget(Arr.RADARR)], config, explicit=True, config_path="config.yml")
        assert kept is None
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert "set radarr.api_key in config.yml" in error.message

    def test_explicit_selection_of_an_unconfigured_arr_refuses(self, recording: RecordingHub) -> None:
        kept = configured_arrs(
            [ArrTarget(Arr.RADARR, 42)],
            _config_with(sonarr=True),
            explicit=True,
            config_path="config.yml",
        )
        assert kept is None
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert "radarr.url" in error.message
        # Prose subject capitalized. The dotted keys above stay lowercase.
        assert error.message.startswith("Radarr was selected but is not configured")

    def test_nothing_configured_reports_and_refuses(self, recording: RecordingHub) -> None:
        kept = configured_arrs(
            [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)],
            _config_with(),
            explicit=False,
            config_path="config.yml",
        )
        assert kept is None
        # Both per-arr skip notes precede the refusal. The ERROR is the verdict.
        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == (
            "Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, or radarr.url and "
            "radarr.api_key, in config.yml"
        )

    def test_fully_configured_passes_through_unchanged(self, recording: RecordingHub) -> None:
        arrs: list[ArrTarget] = [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR, 7)]
        kept = configured_arrs(arrs, _config_with(sonarr=True, radarr=True), explicit=True, config_path="config.yml")
        assert kept == arrs
        assert recording.of_type(Diagnostic) == []


# Mapping stanza shared by the runnable-config helpers: disabled sources keep
# the resolver off the network, so a run gets past the shared-deps stage and
# into the per-arr loop without any I/O.
_NO_MAPPINGS = "mappings:\n  anime_mappings: false\n  anidb_mappings: false\n  anibridge_mappings: false\n"


def _write_runnable_config() -> None:
    """A config with Sonarr configured and every mapping source disabled."""

    paths = resolve_paths()
    os.makedirs(paths.data_dir)
    Path(paths.config).write_text("sonarr:\n  url: http://sonarr:8989\n  api_key: k\n" + _NO_MAPPINGS, encoding="utf-8")


class TestRunFailuresAreCleanAndNonzero:
    """Every failed run leg returns False (exit 1) with a one-liner, not a traceback."""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            pytest.param("qbit", "qBittorrent connection failed", id="qbit-connection"),
            pytest.param(
                "unreachable",
                # The failing URL (from the error's message) is surfaced, so the
                # user sees which connection broke, not just which leg was running.
                "Sonarr run failed - Could not reach Sonarr at http://sonarr:8989 "
                "(request failed (ConnectError)) - check sonarr.url",
                id="arr-unreachable",
            ),
            pytest.param("unauthorized", "Sonarr rejected the API key - check sonarr.api_key", id="arr-unauthorized"),
            pytest.param(
                "contract",
                # The library answered but validated to nothing: the one-line
                # contract arm, never the traceback arm.
                "Sonarr run failed - none of the 3 SonarrSeries records validated",
                id="arr-contract",
            ),
        ],
    )
    def test_a_failed_arr_leg_reports_cleanly_and_returns_false(
        self,
        error: str,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pearlarr.arr_http import ArrAuthError, ArrConnectionError
        from pearlarr.run_services import QbitConnectionError, RunDeps
        from pearlarr.seadex_types import BoundaryContractError

        exceptions: dict[str, Exception] = {
            "qbit": QbitConnectionError("qBittorrent connection failed - check the host and credentials"),
            "unreachable": ArrConnectionError(
                "Could not reach Sonarr at http://sonarr:8989 (request failed (ConnectError))",
            ),
            "unauthorized": ArrAuthError("Sonarr at http://sonarr:8989 rejected the API key (status code 401)"),
            "contract": BoundaryContractError(
                "none of the 3 SonarrSeries records validated. Refusing to treat it as empty",
            ),
        }

        def fail_build(*args: object, **kwargs: object) -> NoReturn:
            raise exceptions[error]

        monkeypatch.setattr(RunDeps, "build", fail_build)
        _write_runnable_config()

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert expected in out
        assert "Traceback" not in out

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            pytest.param("unreachable", "check sonarr.url / radarr.url", id="arr-unreachable"),
            pytest.param(
                "unauthorized",
                "An arr rejected the API key during the Sonarr run - check sonarr.api_key / radarr.api_key",
                id="arr-unauthorized",
            ),
        ],
    )
    def test_a_sonarr_leg_that_also_contacts_radarr_names_both_key_sets(
        self,
        error: str,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Under ignore_movies_in_radarr a Sonarr leg also builds a Radarr client,
        # so a connection/auth failure there must not be pinned on Sonarr alone: a
        # Radarr outage would otherwise read "check sonarr.url" - confidently wrong.
        from pearlarr.arr_http import ArrAuthError, ArrConnectionError
        from pearlarr.run_services import RunDeps

        exceptions: dict[str, Exception] = {
            "unreachable": ArrConnectionError(
                "Could not reach Radarr at http://radarr:7878 (request failed (ConnectError))",
            ),
            "unauthorized": ArrAuthError("Radarr at http://radarr:7878 rejected the API key (status code 401)"),
        }

        def fail_build(*args: object, **kwargs: object) -> NoReturn:
            raise exceptions[error]

        monkeypatch.setattr(RunDeps, "build", fail_build)
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text(
            "sonarr:\n  url: http://sonarr:8989\n  api_key: k\n  ignore_movies_in_radarr: true\n"
            "radarr:\n  url: http://radarr:7878\n  api_key: k\n" + _NO_MAPPINGS,
            encoding="utf-8",
        )

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert expected in out
        assert "Traceback" not in out

    def test_malformed_yaml_fails_the_run_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The syntax error sits ON the credential line: the report must carry the
        # problem + position (from the error's parts), never str(e)'s source
        # snippet, which would quote the api key back to the console/log.
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text('sonarr:\n  api_key: "hunter2\n', encoding="utf-8")

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert "Unreadable YAML" in out
        assert "line 3" in out  # the position survives, so it is easy to find
        assert "hunter2" not in out
        assert "Traceback" not in out

    def test_a_mapping_source_failure_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import pearlarr.mappings as mappings_mod

        def fail_resolver(*args: object, **kwargs: object) -> NoReturn:
            raise RuntimeError("source down")

        monkeypatch.setattr(mappings_mod, "MappingResolver", fail_resolver)
        _write_runnable_config()

        assert run_single(sonarr=True) is False
        assert "Could not fetch/parse the id-mapping sources" in capsys.readouterr().out

    def test_a_mapping_download_network_failure_reports_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A brand-new user with no network hits the FIRST-ever source download
        # (the refresh path falls open to the cached copy, a first download has
        # none): the downloader translates the httpx failure into an OSError,
        # which must surface as a one-line hint, not a ten-frame traceback.
        import pearlarr.mappings as mappings_mod

        def fail_resolver(*args: object, **kwargs: object) -> NoReturn:
            raise OSError("download failed: ConnectError")

        monkeypatch.setattr(mappings_mod, "MappingResolver", fail_resolver)
        _write_runnable_config()

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert "Could not download the id-mapping sources" in out
        assert "check your network connection" in out
        assert "Traceback" not in out

    def test_an_invalid_config_fails_the_run_listing_the_keys(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("sonar:\n  url: x\n", encoding="utf-8")

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert "Invalid configuration" in out
        assert "sonar" in out


class TestSelectionSettlesBeforeMappingFetch:
    """A run with nothing configured fails fast on the selection check, before any mapping source is fetched."""

    def test_nothing_configured_fails_before_the_mapping_fetch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A run with nothing to do must fail fast on the selection check, not
        # download/parse the (large) mapping sources first.
        def fail_build(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError("mapping sources must not be fetched when nothing can run")

        monkeypatch.setattr("pearlarr.bootstrap.build_resolver", fail_build)
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("", encoding="utf-8")  # valid config, neither arr configured

        assert run_single() is False
        capsys.readouterr()  # swallow the boot banner + error line


class TestConfigInspection:
    """`config validate` / `config show`: report cleanly, never write files."""

    def test_validate_missing_file_points_at_init_and_writes_nothing(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert config_validate() is False
        assert "config init" in capsys.readouterr().err
        assert not os.path.exists(resolve_paths().config)

    def test_validate_lists_the_bad_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("sonar:\n  url: x\n", encoding="utf-8")
        assert config_validate() is False
        err = capsys.readouterr().err
        assert "Invalid configuration" in err
        assert "sonar" in err

    def test_validate_reports_malformed_yaml_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text('sonarr:\n  api_key: "hunter2\n', encoding="utf-8")
        assert config_validate() is False
        captured = capsys.readouterr()
        assert "Unreadable YAML" in captured.err
        # Never str(e): its snippet quotes the offending (credential) line.
        assert "hunter2" not in captured.err + captured.out

    def test_validate_reports_what_a_run_would_use(self, capsys: pytest.CaptureFixture[str]) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("sonarr:\n  url: http://sonarr:8989\n  api_key: k\n", encoding="utf-8")
        assert config_validate() is True
        out = capsys.readouterr().out
        assert "OK" in out
        assert "sonarr:      configured" in out
        assert "radarr:      not configured" in out
        assert "preview mode" in out  # no qbittorrent credentials -> nothing is grabbed

    def test_show_masks_set_secrets_and_keeps_unset_ones_null(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text(
            "sonarr:\n  url: http://sonarr:8989\n  api_key: hunter2\n"
            "notifications:\n  discord_url: https://discord.com/api/webhooks/1/tok\n",
            encoding="utf-8",
        )
        assert config_show() is True
        out = capsys.readouterr().out
        assert "hunter2" not in out
        assert "tok" not in out
        assert "api_key: REDACTED" in out
        assert "discord_url: REDACTED" in out
        # The unset radarr key stays null: "is it even set?" must stay answerable.
        assert "api_key: null" in out

    def test_validate_reports_an_old_schema_with_its_folds(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Valid-but-old is a status fact, not a failure: the run migrates it in
        # memory, and validate names the command that updates the file itself.
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("seadex:\n  private_releases: allow\n", encoding="utf-8")
        assert config_validate() is True
        out = capsys.readouterr().out
        assert "OK" in out
        assert "older config schema" in out
        assert "config migrate" in out
        assert "'allow'" in out

    def test_validate_names_the_missing_half_of_a_connection_pair(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("radarr:\n  url: http://radarr:7878\n", encoding="utf-8")
        assert config_validate() is True  # the file itself is valid
        assert "radarr.api_key is not set" in capsys.readouterr().out

    def test_show_masks_the_options_block_and_url_logins(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text(
            "sonarr:\n  url: http://user:basicpass@sonarr:8989\n  api_key: k\n"
            "qbittorrent:\n"
            "  host: http://qbit:8080\n"
            "  username: qbcanaryuser\n"
            "  password: hunter2\n"
            "  options:\n    REQUESTS_ARGS:\n      headers:\n        Authorization: Bearer sekrit\n",
            encoding="utf-8",
        )
        assert config_show() is True
        out = capsys.readouterr().out
        # The free-form options block can hide credentials under any key name
        # (auth headers, proxy URLs), so every value under it is masked...
        assert "sekrit" not in out
        assert "REQUESTS_ARGS: REDACTED" in out
        # ...as are qBittorrent credentials and a login embedded in a URL (the
        # host survives, so the dump still shows where the arr points).
        # The username canary must not collide with legitimate output - a
        # literal "admin" false-positives on Windows CI's runneradmin paths.
        assert "hunter2" not in out
        assert "qbcanaryuser" not in out
        assert "basicpass" not in out
        assert "url: http://REDACTED@sonarr:8989" in out

    def test_show_missing_file_fails_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The failure hint is stderr-only: `config show > cfg.yml` must not
        # capture error text as if it were config.
        assert config_show() is False
        captured = capsys.readouterr()
        assert "config init" in captured.err
        assert captured.out == ""


class TestConfigMigrate:
    """`config migrate`: rewrites an old-schema file behind a backup. Current files untouched."""

    def test_migrates_an_old_file_keeping_values_and_a_backup(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        text = "seadex:\n  public_only: true\nsonarr:\n  url: http://s\n  api_key: hunter2\n"
        Path(paths.config).write_text(text, encoding="utf-8")

        assert config_migrate() is True

        out = capsys.readouterr().out
        assert "previous file saved as" in out
        assert "public_only" in out
        # The command echoes paths and notes, never config values.
        assert "hunter2" not in out
        assert Path(paths.config + ".bak").read_text(encoding="utf-8") == text
        migrated = Path(paths.config).read_text(encoding="utf-8")
        assert "api_key: hunter2" in migrated
        assert "public_only" not in migrated
        assert f"config_version: {CONFIG_VERSION}" in migrated

    def test_a_current_file_is_nothing_to_do(self, capsys: pytest.CaptureFixture[str]) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        text = f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://s\n"
        Path(paths.config).write_text(text, encoding="utf-8")

        assert config_migrate() is True

        assert "nothing to do" in capsys.readouterr().out
        assert Path(paths.config).read_text(encoding="utf-8") == text
        assert not os.path.exists(paths.config + ".bak")

    def test_missing_file_points_at_init_and_writes_nothing(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert config_migrate() is False
        assert "config init" in capsys.readouterr().err
        assert not os.path.exists(resolve_paths().config)

    def test_an_invalid_file_reports_and_stays_untouched(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        text = "seadex:\n  public_only: true\ntypo_key: 1\n"
        Path(paths.config).write_text(text, encoding="utf-8")

        assert config_migrate() is False

        err = capsys.readouterr().err
        assert "Invalid configuration" in err
        assert "typo_key" in err
        assert Path(paths.config).read_text(encoding="utf-8") == text
        assert not os.path.exists(paths.config + ".bak")


class TestVersionAndHelp:
    """`--version`, `-h`, and a bare subcommand group all succeed instead of erroring.

    The flag prints the package version, `-h` aliases `--help`, and the bare group falls back to its help text.
    """

    def test_version_flag_prints_the_package_version(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("pearlarr ")

    def test_short_help_alias_works(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["-h"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_bare_group_shows_its_help(self) -> None:
        # no_args_is_help: a bare `pearlarr run` teaches instead of erroring opaquely.
        result = CliRunner().invoke(pearlarr_cli, ["run"])
        assert "scheduled" in result.output
        assert "single" in result.output


class TestApplyLogLevel:
    """`apply_log_level` re-points the logger and its rich console threshold, including CRITICAL and DEBUG."""

    def test_repoints_logger_and_console_thresholds(self) -> None:
        # Forced rich: "auto" resolves to plain under pytest's non-TTY stdout
        # (the plain twin of this pin lives in test_log_format.py).
        logger = setup_logger(log_level="INFO", console_format="rich")
        console = next(h for h in logger.handlers if isinstance(h, RichConsoleHandler))

        # Raising the level quiets the file log but the console keeps INFO+
        # (routine progress stays visible)...
        apply_log_level(logger, "ERROR")
        assert logger.level == logging.ERROR
        assert console.level == logging.INFO

        # ...while DEBUG and CRITICAL move the console threshold with the logger.
        apply_log_level(logger, "DEBUG")
        assert logger.level == logging.DEBUG
        assert console.level == logging.DEBUG
        apply_log_level(logger, "CRITICAL")
        assert console.level == logging.CRITICAL


class TestLogLevelWiring:
    """The headline fix: `advanced.log_level` reaches CLI runs, `--log-level` wins.

    `apply_log_level` itself is pinned above. These pin that `run_arrs`
    actually calls it once the config is readable, with cli > config precedence
    (the original bug: the config level was dead on every CLI run).
    """

    @pytest.fixture
    def applied(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def record(logger: logging.Logger, log_level: str) -> None:
            calls.append(log_level)

        monkeypatch.setattr("pearlarr.bootstrap.apply_log_level", record)
        return calls

    @staticmethod
    def _write_config_with_level() -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("advanced:\n  log_level: ERROR\n", encoding="utf-8")

    def test_the_config_level_is_applied_to_a_run(
        self,
        applied: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._write_config_with_level()
        # Nothing is configured, so the run refuses - but only after the level
        # is applied (the refusal itself must respect the configured level).
        assert run_single() is False
        assert applied == ["ERROR"]
        capsys.readouterr()  # swallow the boot banner + refusal line

    def test_the_cli_override_wins_over_the_config(
        self,
        applied: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._write_config_with_level()
        assert run_single(log_level=LogLevel.DEBUG) is False
        assert applied == [LogLevel.DEBUG]
        capsys.readouterr()


class _SetupLoggerRecorder:
    """A stand-in for `cli.setup_logger` recording each call's context.

    Captures the console_format it got and whether the output bridge was
    already installed on the app logger AT CALL TIME (the install-order pin:
    a record fired from inside setup_logger must reach the hub).
    """

    def __init__(self) -> None:
        self.console_formats: list[LogFormat] = []
        self.bridge_installed_at_call: list[bool] = []

    def __call__(self, log_level: str, console_format: LogFormat = "auto") -> logging.Logger:
        self.console_formats.append(console_format)
        self.bridge_installed_at_call.append(
            any(isinstance(h, HubBridgeBase) for h in logging.getLogger(LOG_NAME).handlers),
        )
        return make_logger()


class _StopScheduledLoop(Exception):
    """Breaks `run_scheduled`'s infinite loop from a faked `run_arrs`."""


def _stop_loop(*args: object, **kwargs: object) -> NoReturn:
    """A `run_arrs` stand-in that breaks `run_scheduled`'s loop."""

    raise _StopScheduledLoop


def _swallow_signal(signum: int, handler: object) -> None:
    """A no-op `signal.signal`: a test must never re-point the pytest process's real SIGTERM disposition.

    The registration itself is pinned separately.
    """


class _CycleStampingRenderer(Renderer):
    """Records each hub event alongside how many `begin_cycle` calls preceded it.

    Makes post-begin ordering assertable (CycleStarted lands in the fresh cycle).
    """

    writes_file_only: ClassVar[bool] = False

    def __init__(self) -> None:
        self.cycles = 0
        self.events: list[tuple[Event, int]] = []

    @override
    def handle(self, event: Event, when: float) -> None:
        self.events.append((event, self.cycles))

    @override
    def begin_cycle(self) -> None:
        self.cycles += 1

    @override
    def set_level(self, level: int) -> None:
        pass

    @override
    def close(self) -> None:
        pass


def _install_cycle_recorder(monkeypatch: pytest.MonkeyPatch) -> _CycleStampingRenderer:
    """Swap `cli._install_output_hub` for a hub over one recording renderer."""

    renderer = _CycleStampingRenderer()

    def install(paths: AppPaths) -> tuple[OutputHub, FileLogSink]:
        hub = OutputHub([renderer])
        install_hub(hub)  # emit_to_hub resolves this hub, conftest restores
        return hub, FileLogSink(paths.log_dir)

    monkeypatch.setattr("pearlarr.cli._install_output_hub", install)
    return renderer


class TestLogFormatWiring:
    """`advanced.log_format` reaches `setup_logger` on both run commands.

    The renderers themselves are pinned in test_log_format.py. These pin that
    the run commands resolve the config format and thread `console_format`
    through (scheduled mode re-resolves each cycle, before the logger is
    rebuilt), AND that the hub + bridge are installed BEFORE setup_logger runs
    (advisor #17: an invalid-level complaint fired inside it must reach the
    hub, never logging.lastResort).
    """

    @pytest.fixture
    def setup_recorder(self, monkeypatch: pytest.MonkeyPatch) -> _SetupLoggerRecorder:
        recorder = _SetupLoggerRecorder()
        monkeypatch.setattr("pearlarr.cli.setup_logger", recorder)
        return recorder

    @staticmethod
    def _write_config_with_format() -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("advanced:\n  log_format: json\n", encoding="utf-8")

    def test_run_single_threads_the_config_format(
        self,
        setup_recorder: _SetupLoggerRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", _RunArrsRecorder())
        self._write_config_with_format()
        assert run_single() is True
        assert setup_recorder.console_formats == ["json"]
        assert setup_recorder.bridge_installed_at_call == [True]  # hub+bridge first

    def test_run_scheduled_threads_the_config_format(
        self,
        setup_recorder: _SetupLoggerRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        monkeypatch.setattr(signal, "signal", _swallow_signal)
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", _stop_loop)
        self._write_config_with_format()
        with pytest.raises(_StopScheduledLoop):
            run_scheduled()
        assert setup_recorder.console_formats == ["json"]
        assert setup_recorder.bridge_installed_at_call == [True]  # installed pre-loop


class TestConsoleFormat:
    """`_console_format`: a silent pre-logger peek folding every failure to "auto"."""

    def test_missing_config_folds_to_auto_and_writes_nothing(self, tmp_path: Path) -> None:
        missing = tmp_path / "config.yml"
        assert _console_format(str(missing)) == "auto"
        # No template-copy side effect: load_shared_config owns the first-run copy.
        assert not missing.exists()

    def test_reads_the_configured_format(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yml"
        # Case-folded by the config validator, like log_level.
        path.write_text("advanced:\n  log_format: JSON\n", encoding="utf-8")
        assert _console_format(str(path)) == "json"

    def test_invalid_config_folds_to_auto(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yml"
        path.write_text("advanced:\n  log_format: fancy\n", encoding="utf-8")
        assert _console_format(str(path)) == "auto"

    def test_malformed_yaml_folds_to_auto(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yml"
        path.write_text("advanced: [unclosed\n", encoding="utf-8")
        assert _console_format(str(path)) == "auto"


class TestResolvedFormat:
    """`_resolved_format` is the one "auto" fold.

    The same resolved value feeds `setup_logger` and `hub.begin_cycle`, so the two can never disagree.
    """

    def test_auto_folds_to_plain_off_a_tty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        assert _resolved_format(str(tmp_path / "config.yml")) == "plain"  # missing config -> auto -> plain

    def test_auto_folds_to_rich_on_a_tty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdout", TtyStringIO())
        assert _resolved_format(str(tmp_path / "config.yml")) == "rich"

    def test_a_configured_format_passes_through_unfolded(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yml"
        path.write_text("advanced:\n  log_format: json\n", encoding="utf-8")
        assert _resolved_format(str(path)) == "json"


def _drive_representative(hub: OutputHub) -> None:
    """Boot facts, scan lines, both diagnostic classes, a graduation, a summary."""

    hub.emit(RunStarted(version="v9.9.9", data_dir="/data"))
    hub.emit(ScanStarted(arr=Arr.SONARR, total=1))
    hub.emit(ItemStarted(arr=Arr.SONARR, index=1, total=1, title="Frieren"))
    hub.emit(Diagnostic(severity=Severity.WARNING, message="tracker down", origin=LOG_NAME))
    hub.emit(Diagnostic(severity=Severity.WARNING, message="forensic note", origin="output.hub", file_only=True))
    hub.emit(TorrentGraduated(label="Show", outcome=Outcome.DOWNLOAD_ERRORED, files=None, waited_s=60.0))
    hub.emit(SUMMARY_MINIMAL)


class TestHubSeats:
    """The REAL cli seat factory, pinned end-to-end.

    Plain stdout is the file's bytes minus exactly the file_only lines. json is
    event-per-line.
    """

    @staticmethod
    def _seated(
        log_dir: Path,
        console_format: LogFormat,
        level: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[OutputHub, io.StringIO]:
        stream = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stream)
        hub = OutputHub(
            [FileLogSink(str(log_dir))],
            console_factory=partial(_console_seat, caps_cache=CapsCache()),
        )
        hub.begin_cycle(console_format=console_format, level=level)
        return hub, stream

    @pytest.mark.parametrize("level", [logging.INFO, logging.WARNING])
    def test_plain_stdout_is_the_file_minus_exactly_the_file_only_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        level: int,
    ) -> None:
        hub, stream = self._seated(tmp_path / "logs", "plain", level, monkeypatch)
        _drive_representative(hub)
        hub.close()

        stdout_lines = stream.getvalue().splitlines()
        file_lines = (tmp_path / "logs" / "Pearlarr.log").read_text(encoding="utf-8").splitlines()
        assert stdout_lines  # WARNING still keeps the diagnostic + the ERROR graduation
        assert [line for line in file_lines if "forensic note" not in line] == stdout_lines
        assert sum("forensic note" in line for line in file_lines) == 1  # the sole carve-out

    def test_json_stdout_is_one_json_object_per_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hub, stream = self._seated(tmp_path / "logs", "json", logging.INFO, monkeypatch)
        _drive_representative(hub)
        hub.close()

        events = [cast("dict[str, object]", json.loads(line)) for line in stream.getvalue().splitlines()]
        envelope = ["schema_version", "time", "event", "level", "message"]
        assert [list(obj)[:5] for obj in events] == [envelope] * len(events)
        # One object per EVENT, in emit order. The file_only diagnostic stays out.
        assert [obj["event"] for obj in events] == [
            "run_started",
            "scan_started",
            "item_started",
            "diagnostic",
            "torrent_graduated",
            "run_summary",
        ]


class TestMissingConfigExitsNonzero:
    """A virgin data dir: exit 1 with the starter template written, never a silent retry.

    Driven end-to-end through CliRunner: the `typer.Exit` raised inside
    `load_shared_config`'s FileNotFoundError arm must escape `run_arrs`'
    finallys (boot view, web client, run lock all release) and reach typer as
    exit code 1 - pinning the whole propagation path, not just the helper.
    """

    def test_run_single_writes_the_template_and_exits_one(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1

        config = Path(resolve_paths().config)
        assert config.read_text(encoding="utf-8") == _starter_text()
        assert "starter template was written" in result.output
        # This arm exits now. The skip-and-retry wording belongs to the others.
        assert "Skipping this run" not in result.output


class TestUnknownTrackerWarning:
    """Unknown `seadex.trackers` values warn at load - they silently match nothing.

    Warn-not-reject: strict key validation already rejects unknown KEYS. An unknown
    VALUE is a typo that would quietly filter out every release from that tracker.
    """

    @staticmethod
    def _load(tmp_path: Path, body: str) -> AppConfig | None:
        config = tmp_path / "config.yml"
        # Current-schema stamp + 0600: keep the old-schema and loose-permissions
        # warnings out of the recorded stream (this class is about trackers).
        config.write_text(f"config_version: {CONFIG_VERSION}\n{body}", encoding="utf-8")
        config.chmod(0o600)
        return load_shared_config(str(config), BootFlow(), "")

    def test_unknown_value_warns_naming_it_and_the_vocabulary(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        assert self._load(tmp_path, "seadex:\n  trackers: [Nyaaa, Nyaa]\n") is not None

        warning = next(d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING)
        assert "nyaaa" in warning.message  # the offender (casefolded, as matching sees it)
        assert "case-insensitive" in warning.message
        assert "animetosho" in warning.message  # the known vocabulary is enumerated
        assert "otherprivate" in warning.message  # including the seadex-only catch-alls

    def test_known_values_do_not_warn(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        loaded = self._load(tmp_path, "seadex:\n  trackers: [Nyaa, OtherPrivate]\n")

        assert loaded is not None
        assert loaded.seadex.trackers == {"nyaa", "otherprivate"}
        assert [d for d in recording.of_type(Diagnostic) if d.severity >= Severity.WARNING] == []

    def test_default_trackers_do_not_warn(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        assert self._load(tmp_path, "sonarr:\n  url: http://s\n") is not None
        assert [d for d in recording.of_type(Diagnostic) if d.severity >= Severity.WARNING] == []


class TestUnexpectedConfigErrorSkips:
    """`load_shared_config`'s last-resort arm: ERROR with traceback, then skip."""

    def test_unexpected_load_error_reports_and_skips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No file content reaches this arm (undecodable bytes are a YAMLError,
        # bad keys a ValidationError), so the "impossible" failure is injected.
        def boom(path: str) -> AppConfig:
            raise RuntimeError(f"config loader broke on {path}")

        monkeypatch.setattr(AppConfig, "load", boom)
        recording = install_recording_hub()
        config = str(tmp_path / "config.yml")

        assert load_shared_config(config, BootFlow(), " - will retry") is None

        (error,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]
        assert error.message == f"Could not load config {config} - skipping this run - will retry"
        assert error.trace is not None  # unexpected -> the traceback is kept


# chmod can't produce a read-only DIRECTORY or an unreadable file on Windows
# (only the file read-only attribute binds), so these scenarios need POSIX.
_needs_posix_permissions = pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod-based access denial does not bind on Windows",
)


class TestUnwritableDataDir:
    """An unwritable data directory: one actionable stderr line + exit 1, no traceback.

    `ensure_data_dir`/`setup_logger` run before any logger exists, so the
    report must go straight to stderr. `config init`'s template copy gets the
    same treatment. An unreadable CONFIG in a healthy dir instead keeps the
    skip+retry contract - exiting would kill the scheduled daemon.
    """

    @_needs_posix_permissions
    def test_run_single_reports_and_exits_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ro_parent = tmp_path / "ro"
        ro_parent.mkdir()
        data_dir = ro_parent / "data"
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(data_dir))
        ro_parent.chmod(0o500)
        try:
            result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        finally:
            ro_parent.chmod(0o755)

        assert result.exit_code == 1
        assert f"Cannot write to the data directory {data_dir}" in result.stderr
        assert "PEARLARR_DATA_DIR" in result.stderr  # the remedy is named
        assert "Traceback" not in result.output

    def test_an_unwritable_log_file_reports_and_exits_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The Docker shape: dir writable, Pearlarr.log left root-owned by a prior
        # run. The install-time probe must abort cleanly, not strike the sink.
        data_dir = tmp_path / "data"
        log = data_dir / "logs" / "Pearlarr.log"
        log.parent.mkdir(parents=True)
        log.write_text("root-owned\n", encoding="utf-8")
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(data_dir))
        log.chmod(0o400)
        try:
            result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        finally:
            log.chmod(0o644)

        assert result.exit_code == 1
        assert f"Cannot write to the data directory {data_dir}" in result.stderr
        assert "Traceback" not in result.output

    @_needs_posix_permissions
    def test_config_init_into_a_readonly_dir_reports_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The dir EXISTS read-only: makedirs(exist_ok=True) passes and the
        # template copy is what fails - it needs the same one-line treatment.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(data_dir))
        data_dir.chmod(0o500)
        try:
            result = CliRunner().invoke(pearlarr_cli, ["config", "init"])
        finally:
            data_dir.chmod(0o755)

        assert result.exit_code == 1
        assert f"Cannot write to the data directory {data_dir}" in result.stderr
        assert "Traceback" not in result.output

    @_needs_posix_permissions
    def test_an_unreadable_config_skips_the_run_instead_of_exiting(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arm order in load_shared_config matters twice: FileNotFoundError
        # (exit 1) must not swallow this OSError, and the OSError arm must
        # catch it before the traceback arm. Skip + retry, never exit.
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        config = Path(paths.config)
        config.write_text("sonarr:\n  url: http://s\n  api_key: k\n", encoding="utf-8")
        config.chmod(0o000)
        try:
            assert run_single(sonarr=True) is False
        finally:
            config.chmod(0o644)

        out = capsys.readouterr().out
        assert "Could not access config" in out
        assert "skipping this run" in out
        assert "Traceback" not in out


class TestDataDirLine:
    """Every run states the resolved data directory on its opening line."""

    def test_run_single_logs_the_resolved_data_dir(self) -> None:
        # Virgin data dir: the run exits 1 writing the template, but the
        # run_started line (plain seat, structured grammar) carries the
        # data_dir fact FIRST, before the config read fails.
        result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1
        assert "Pearlarr started" in result.output
        assert f"data_dir={resolve_paths().data_dir}" in result.output


class TestScheduledLifecycle:
    """Scheduled mode: invalid configs retry, SIGTERM exits 0, help names the fallback."""

    def test_invalid_config_still_skips_and_retries(self, logger: logging.Logger) -> None:
        # Only the MISSING file exits: an invalid config is likely mid-edit, so
        # scheduled mode must keep skipping + retrying, never raise out of the loop.
        recording = install_recording_hub()
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("sonar:\n  url: x\n", encoding="utf-8")

        result = run_arrs(
            [ArrTarget(Arr.SONARR)],
            paths=paths,
            logger=logger,
            file_sink=FileLogSink(paths.log_dir),
            retry_note="will retry in 6h (Ctrl-C to stop)",
        )

        assert result is False
        error = next(d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR)
        assert "will retry in 6h" in error.message

    def test_an_old_schema_config_warns_and_still_loads(
        self,
        logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The in-memory migration keeps an old file running. The warn names what
        # was folded and the command that updates the file. The resolver stub
        # stops the run right after the config load (no network).
        recording = install_recording_hub()
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text(
            "seadex:\n  private_releases: allow\nsonarr:\n  url: http://s\n  api_key: k\n",
            encoding="utf-8",
        )

        def refuse_resolver(*args: object, **kwargs: object) -> None:
            return None

        monkeypatch.setattr("pearlarr.bootstrap.build_resolver", refuse_resolver)

        assert (
            run_arrs([ArrTarget(Arr.SONARR)], paths=paths, logger=logger, file_sink=FileLogSink(paths.log_dir)) is False
        )

        # Selected by content: the 0644 write also draws the loose-permissions warn.
        warning = next(d for d in recording.of_type(Diagnostic) if "older config schema" in d.message)
        assert warning.severity is Severity.WARNING
        assert "'allow'" in warning.message
        assert "pearlarr config migrate" in warning.message

    def test_run_arrs_applies_the_configured_log_retention(
        self,
        logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The knob must reach the sink with the CONFIGURED value, not a default.
        # the resolver stub stops the run right after (no network).
        install_recording_hub()
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text(
            "sonarr:\n  url: http://s\n  api_key: k\nadvanced:\n  log_retention_days: 3\n",
            encoding="utf-8",
        )
        applied: list[int] = []
        original = FileLogSink.apply_retention_days

        def record(self: FileLogSink, days: int) -> None:
            applied.append(days)
            original(self, days)

        def refuse_resolver(*args: object, **kwargs: object) -> None:
            return None

        monkeypatch.setattr(FileLogSink, "apply_retention_days", record)
        monkeypatch.setattr("pearlarr.bootstrap.build_resolver", refuse_resolver)

        run_arrs([ArrTarget(Arr.SONARR)], paths=paths, logger=logger, file_sink=FileLogSink(paths.log_dir))

        assert applied == [3]

    def test_sigterm_handler_emits_the_hub_notice_and_exits_zero(self) -> None:
        # Call the handler directly - never deliver a real signal in-process.
        recording = install_recording_hub()
        with pytest.raises(SystemExit) as excinfo:
            _handle_sigterm(signal.SIGTERM, None)

        assert excinfo.value.code == 0
        (note,) = recording.of_type(Diagnostic)
        assert note.severity is Severity.INFO
        assert note.message == "Received SIGTERM - exiting"
        assert note.origin == LOG_NAME

    def test_cycle_started_is_emitted_after_begin_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Post-begin (rotation): the boundary must land in the FRESH cycle's file.
        monkeypatch.setattr(signal, "signal", _swallow_signal)
        monkeypatch.setattr("pearlarr.cli.setup_logger", _SetupLoggerRecorder())
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", _stop_loop)
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        renderer = _install_cycle_recorder(monkeypatch)

        with pytest.raises(_StopScheduledLoop):
            run_scheduled()

        started = [(e, cycles) for e, cycles in renderer.events if isinstance(e, CycleStarted)]
        assert [(e.number, cycles) for e, cycles in started] == [(1, 1)]

    def test_next_run_scheduled_is_emitted_with_the_cadence_ahead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # After a completed cycle the loop states the next run as a typed event
        # (default cadence: 6h out on a missing config), then sleeps.
        monkeypatch.setattr(signal, "signal", _swallow_signal)
        monkeypatch.setattr("pearlarr.cli.setup_logger", _SetupLoggerRecorder())
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", _RunArrsRecorder())
        monkeypatch.setattr("pearlarr.cli.time.sleep", _stop_loop)
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        renderer = _install_cycle_recorder(monkeypatch)

        before = datetime.now().astimezone()
        with pytest.raises(_StopScheduledLoop):
            run_scheduled()

        (scheduled,) = [e for e, _ in renderer.events if isinstance(e, NextRunScheduled)]
        assert scheduled.at.tzinfo is not None  # aware: the serialized form carries its offset
        assert abs((scheduled.at - (before + timedelta(hours=6))).total_seconds()) < 60

    def test_run_scheduled_registers_the_sigterm_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[tuple[int, object]] = []

        def record_signal(signum: int, handler: object) -> None:
            registered.append((signum, handler))

        monkeypatch.setattr(signal, "signal", record_signal)
        monkeypatch.setattr("pearlarr.cli.setup_logger", _SetupLoggerRecorder())
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", _stop_loop)
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)

        with pytest.raises(_StopScheduledLoop):
            run_scheduled()

        assert registered == [(signal.SIGTERM, _handle_sigterm)]

    def test_scheduled_help_names_the_bare_metal_fallback(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "scheduled", "--help"])
        assert result.exit_code == 0
        assert "bare-metal" in result.output


class TestRunLockContention:
    """A second run against the same data dir warns once and returns False."""

    def test_run_arrs_skips_when_the_lock_is_already_held(self, logger: logging.Logger) -> None:
        recording = install_recording_hub()
        paths = resolve_paths()
        os.makedirs(paths.data_dir)

        # Hold the run lock ourselves. run_arrs must find it taken and bail
        # before ever reading the config (none exists here - a regression past
        # the guard would raise the missing-config typer.Exit).
        with single_instance_lock(paths.data_dir) as held:
            assert held
            result = run_arrs([ArrTarget(Arr.SONARR)], paths=paths, logger=logger, file_sink=FileLogSink(paths.log_dir))

        assert result is False
        assert diagnostic_messages(recording, Severity.WARNING) == [
            f"Another Pearlarr run is active in {paths.data_dir} - skipping this run",
        ]


class TestScheduleHours:
    """Cadence precedence: valid SCHEDULE_TIME env (deprecated) > config > 6.

    The SCHEDULE_TIME notices are first-party WARNING Diagnostics through the
    hub (file line + console badge). Pinned here via a recording hub.
    """

    @pytest.fixture
    def recording(self) -> RecordingHub:
        return install_recording_hub()

    @staticmethod
    def _write_config(tmp_path: Path, interval_hours: float | None = None) -> str:
        path = tmp_path / "config.yml"
        body = "" if interval_hours is None else f"schedule:\n  interval_hours: {interval_hours}\n"
        path.write_text(body)
        return str(path)

    def test_unset_env_and_missing_config_uses_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recording: RecordingHub,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        missing = tmp_path / "config.yml"
        assert _schedule_hours(str(missing)) == 6.0
        # No template-copy side effect: load_shared_config owns the first-run copy.
        assert not missing.exists()
        assert recording.of_type(Diagnostic) == []  # nothing to warn about without the env var

    def test_unset_env_reads_the_config_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        assert _schedule_hours(self._write_config(tmp_path, 2.5)) == 2.5

    def test_unreadable_config_falls_back_to_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        path = tmp_path / "config.yml"
        path.write_text("schedule:\n  interval_hours: -1\n")  # fails validation
        assert _schedule_hours(str(path)) == 6.0

    def test_a_valid_env_value_wins_over_config_with_deprecation_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recording: RecordingHub,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", "0.5")
        assert _schedule_hours(self._write_config(tmp_path, 2.5)) == 0.5
        (notice,) = recording.of_type(Diagnostic)
        assert notice.severity is Severity.WARNING
        assert notice.message == "SCHEDULE_TIME is deprecated - set schedule.interval_hours in the config instead"

    @pytest.mark.parametrize("raw", ["banana", "0", "-3", "inf", "nan"])
    def test_bad_env_values_fall_through_to_config(
        self,
        raw: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recording: RecordingHub,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", raw)
        assert _schedule_hours(self._write_config(tmp_path, 2.5)) == 2.5
        (notice,) = recording.of_type(Diagnostic)
        assert notice.severity is Severity.WARNING
        assert "Invalid SCHEDULE_TIME" in notice.message
        # The notice reports the value actually used, not a claim about its source.
        assert "using 2.5 hours" in notice.message

    def test_bad_env_with_missing_config_reports_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recording: RecordingHub,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", "banana")
        assert _schedule_hours(str(tmp_path / "config.yml")) == 6.0
        assert any("using 6 hours" in d.message for d in recording.of_type(Diagnostic))

    def test_malformed_yaml_falls_back_to_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        path = tmp_path / "config.yml"
        path.write_text("schedule: [unclosed\n")
        assert _schedule_hours(str(path)) == 6.0


def test_trust_os_certificates_swaps_in_the_os_trust_store() -> None:
    """The root callback's TLS hook swaps `ssl.SSLContext` for truststore's OS-backed one.

    It runs before any HTTP client builds a context.
    """

    try:
        _trust_os_certificates()
        assert ssl.SSLContext is truststore.SSLContext
    finally:
        truststore.extract_from_ssl()


class TestExitCodes:
    """The result callback maps a False return to exit code 1 (typer ignores returns)."""

    def test_failure_exits_nonzero_and_success_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        runner = CliRunner()

        # No cache.db yet: the command reports the missing file (on stderr,
        # keeping stdout clean for scripts) and must exit 1.
        result = runner.invoke(pearlarr_cli, ["cache", "stats"])
        assert result.exit_code == 1
        assert "No cache database at" in result.stderr
        assert result.stdout == ""

        _build_cache(tmp_path)
        result = runner.invoke(pearlarr_cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "entries:" in result.stdout


def test_cache_stats_pins_the_padded_human_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The renderer reproduces the pre-hub typer.echo block byte-for-byte:
    # 17-column padded labels and a two-decimal MiB size.
    monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
    _build_cache(tmp_path)

    result = CliRunner().invoke(pearlarr_cli, ["cache", "stats"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[:5] == [
        "entries:         1",
        "torrent_hashes:  0",
        "anilist_meta:    0",
        "sonarr_parse:    0",
        "pending_imports: 0",
    ]
    assert re.fullmatch(r"size:            \d+\.\d\d MiB", lines[5])
    assert len(lines) == 6
