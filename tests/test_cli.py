# pyright: strict
# pyright: reportPrivateUsage=false
# A few tests reach private helpers. The repo already disables reportPrivateUsage
# for all of tests/, but the strict directive above re-enables it, so restore the
# repo's test policy here.
"""Tests for the Pearlarr CLI commands.

Pins the behaviors the commands must guarantee:

* A corrupt / not-a-database `cache.db` is *reported*, never crashed on - the
  `stats` and `check` diagnostics (and `backup`) return a clean False instead
  of letting a `sqlite3` traceback escape. `check` exists to report
  bad integrity, so it must survive the very corruption it diagnoses.
* The destructive commands (`restore` / `remove`) take the single-instance run
  lock first and refuse while a run is active, so they never unlink or replace the
  live db out from under it.
* Failure paths report cleanly (missing files echo one hint line, not a traceback),
  failure text goes to stderr (so `pearlarr config show > cfg.yml` stays clean)
  while success output stays on stdout, and a False return maps to exit code 1
  through the Typer apps' result callback.
* `config init` never overwrites a filled-in config.yml without `--force`.
* `run single` with no selection flag runs every *configured* arr (scheduled-mode
  symmetry). An explicit flag for an unconfigured arr refuses cleanly, an implicit
  selection skips it with a dim note in the boot step lines - except a half-configured arr (url
  without api_key), whose skip warns by name (`configured_arrs`).
* The inspection commands (`config validate` / `config show`) never write the
  template, and `show` masks secret-named values (plus the free-form
  `qbittorrent.options` block and URL-embedded logins) while keeping unset
  ones `null`.
* `advanced.log_level` is applied to CLI runs as soon as the config is read,
  and `--log-level` overrides it (cli > config).
* A missing config exits 1 with the template written (no silent
  skip-and-retry). An invalid config keeps the skip+retry contract in scheduled
  mode. SIGTERM stops either run command with exit code 0.

Each test points `resolve_paths()` at its own `tmp_path` via `PEARLARR_DATA_DIR`
and calls the command functions directly (they return `bool`). The exit-code
tests go through `CliRunner` since the callback only runs inside typer.
"""

import json
import logging
import os
import re
import signal
import ssl
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn, cast

import httpx
import pytest
import respx
import truststore
from typer.testing import CliRunner, Result

from pearlarr.boot_flow import BootFlow
from pearlarr.bootstrap import _rewrite_old_config, configured_arrs, load_shared_config, run_arrs
from pearlarr.cache import CacheStore
from pearlarr.cli import (
    _handle_sigterm,
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
    run_single,
)
from pearlarr.config import AppConfig, Arr, ArrTarget, template_path
from pearlarr.config_migrations import CONFIG_VERSION
from pearlarr.log import LOG_NAME
from pearlarr.output import (
    Diagnostic,
    FileLogRenderer,
    Severity,
    current_hub,
)
from pearlarr.paths import AppPaths, resolve_paths
from pearlarr.runlock import single_instance_lock
from pearlarr.wait_types import ImportWaitMode

from .builders import write_cache_db
from .fakes import RecordingHub, install_recording_hub


def _starter_text() -> str:
    """What a user's starter copy holds: the template minus its generated-file banner."""

    template = Path(template_path()).read_text(encoding="utf-8")
    return "".join(line for line in template.splitlines(keepends=True) if not line.startswith("# GENERATED"))


def _write_data_dir_config(body: str) -> Path:
    """Write `body` as the config in the test's data dir, where `resolve_paths` looks for it, and give its path."""

    paths = resolve_paths()
    os.makedirs(paths.data_dir)
    config = Path(paths.config_file)
    config.write_text(body, encoding="utf-8")
    return config


def _write_current_config(body: str = "") -> Path:
    """`body` as a current-schema, owner-only config, so a run adds no migration or permissions line of its own."""

    config = _write_data_dir_config(f"config_version: {CONFIG_VERSION}\n{body}")
    config.chmod(0o600)
    return config


_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")


def _lines(text: str) -> list[str]:
    """A run's plain console or log file lines, without the timestamp each one starts with."""

    return [_TIMESTAMP.sub("", line) for line in text.splitlines()]


# The notes a run that asked for both arrs prints when neither is configured.
_SKIP_NOTES = [
    "INFO [Pearlarr] Radarr not configured - not run during=boot",
    "INFO [Pearlarr] Sonarr not configured - not run during=boot",
]


def _nothing_configured_line(config: Path) -> str:
    """The plain line a run refuses with when `config` configures neither arr."""

    return (
        "ERROR [Pearlarr] Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, "
        f"or radarr.url and radarr.api_key, in {config} during=boot"
    )


def _invoke_offline(*args: str) -> Result:
    """Invoke the CLI with every outbound request refused, failing the test if one was made.

    A run that gets past where it should stop then fails here instead of reaching the network.
    """

    with respx.mock(assert_all_called=False) as router:
        web = router.route().mock(side_effect=httpx.ConnectError("refused by the test"))
        result = CliRunner().invoke(pearlarr_cli, list(args))
    assert not web.called
    return result


@pytest.fixture
def recording() -> RecordingHub:
    """A recording hub in place of the default for the test."""

    return install_recording_hub()


class TestCacheRoundTrip:
    """`cache_backup`/`cache_restore`/`cache_remove` round-trip and clean up the cache db.

    Backup-then-restore preserves data and clears a leftover WAL/SHM sidecar. Remove deletes the db and its sidecars.
    """

    def test_backup_then_restore_preserves_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        write_cache_db(tmp_path, series_id=7, name="X")

        assert cache_backup() is True
        assert (tmp_path / "cache.backup.db").exists()
        # Success is confirmed out loud (on stdout), not silently.
        assert f"Backed up cache to {tmp_path / 'cache.backup.db'}" in capsys.readouterr().out

        # A leftover WAL next to cache.db must not shadow the restored snapshot.
        # restore clears the sidecars before swapping the copy into place.
        (tmp_path / "cache.db-wal").write_text("stale")

        assert cache_restore() is True
        assert f"Restored cache from {tmp_path / 'cache.backup.db'}" in capsys.readouterr().out
        # Copy-restore: the backup SURVIVES (a post-restore corruption can be
        # restored again) and no temp file is left behind.
        assert (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.db.tmp").exists()
        assert not (tmp_path / "cache.db-wal").exists()  # leftover sidecar cleared

        # Restore is repeatable off the surviving backup.
        assert cache_restore() is True
        capsys.readouterr()  # drain the second success echo (keeps it off the terminal under -s)

        # The restored db still holds the original cached title.
        store = CacheStore.open_readonly(str(tmp_path / "cache.db"))
        try:
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.name == "X"
        finally:
            store.close()

    def test_remove_deletes_db_and_sidecars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        write_cache_db(tmp_path)
        (tmp_path / "cache.db-wal").write_text("stale")
        (tmp_path / "cache.db-shm").write_text("stale")

        assert cache_remove() is True
        assert f"Removed {tmp_path / 'cache.db'}" in capsys.readouterr().out
        assert not (tmp_path / "cache.db").exists()
        assert not (tmp_path / "cache.db-wal").exists()
        assert not (tmp_path / "cache.db-shm").exists()


class TestHealthyDiagnostics:
    """`cache_check` succeeds on a healthy db and reports its integrity as ok."""

    def test_check_reports_a_healthy_db_as_ok(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        write_cache_db(tmp_path)

        assert cache_check() is True
        assert "integrity: ok" in capsys.readouterr().out


class TestCorruptDatabaseIsReportedNotCrashed:
    """A corrupt cache.db is reported cleanly, never tracebacks."""

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
        write_cache_db(tmp_path, series_id=7, name="X")
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
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.name == "X"
        finally:
            store.close()


class TestActiveRunGuard:
    """Cache commands refuse while a run holds the lock."""

    def test_remove_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        write_cache_db(tmp_path)

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
        write_cache_db(tmp_path)
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
        write_cache_db(tmp_path)

        # A mid-run snapshot would capture half-committed run state. Backup
        # refuses like restore/remove and writes nothing.
        with single_instance_lock(str(tmp_path)):
            assert cache_backup() is False
        assert not (tmp_path / "cache.backup.db").exists()
        assert "another pearlarr run is active" in capsys.readouterr().err.lower()


class TestMissingFilesAreReportedNotRaised:
    """A missing cache/backup file is one hinting stderr line and exit 1, leaving stdout clean for scripts."""

    @pytest.mark.parametrize(
        ("command", "what", "file", "hint"),
        [
            pytest.param("backup", "cache database", "cache.db", "it is created by the first run", id="backup"),
            pytest.param("stats", "cache database", "cache.db", "it is created by the first run", id="stats"),
            pytest.param("check", "cache database", "cache.db", "it is created by the first run", id="check"),
            pytest.param("remove", "cache database", "cache.db", "nothing to remove", id="remove"),
            pytest.param("restore", "backup", "cache.backup.db", "run pearlarr cache backup first", id="restore"),
        ],
    )
    def test_each_cache_command_reports_a_missing_file_with_a_hint(
        self,
        command: str,
        what: str,
        file: str,
        hint: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))

        result = CliRunner().invoke(pearlarr_cli, ["cache", command])

        # The line names what's missing and how to get one, so a fresh install
        # isn't met with a bare "No file at ...".
        assert result.exit_code == 1
        assert result.stderr == f"No {what} at {tmp_path / file} - {hint}\n"
        assert result.stdout == ""


class TestConfigInit:
    """config init writes the template but never clobbers without --force."""

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
    """A fake `bootstrap.run_arrs` that records the arrs and wait mode `run_single` asks it for."""

    def __init__(self) -> None:
        self.arrs: list[tuple[Arr, int | None]] | None = None
        self.import_wait_mode: ImportWaitMode | None = None

    def __call__(
        self,
        arrs: list[tuple[Arr, int | None]],
        *,
        paths: AppPaths,
        logger: logging.Logger,
        file_log_renderer: FileLogRenderer,
        explicit_selection: bool = False,
        dry_run: bool = False,
        import_wait_mode: ImportWaitMode | None = None,
        log_level: str | None = None,
        retry_note: str | None = None,
    ) -> bool:
        self.arrs = arrs
        self.import_wait_mode = import_wait_mode
        return True


class TestRunSingleSelection:
    """No selection flag asks for every arr. A flag or an id narrows the run to its arr.

    `run_arrs` is faked out, so these pin only what the flags ask for. What a
    run does with an arr that isn't configured is pinned on `configured_arrs` below.
    """

    @pytest.fixture
    def recorder(self, monkeypatch: pytest.MonkeyPatch) -> _RunArrsRecorder:
        recorder = _RunArrsRecorder()
        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", recorder)
        return recorder

    @pytest.mark.parametrize(
        ("flags", "arrs"),
        [
            pytest.param([], [(Arr.RADARR, None), (Arr.SONARR, None)], id="no-flag"),
            pytest.param(["--sonarr"], [(Arr.SONARR, None)], id="sonarr"),
            pytest.param(["--radarr"], [(Arr.RADARR, None)], id="radarr"),
            pytest.param(["--tmdb-id", "42"], [(Arr.RADARR, 42)], id="tmdb-id"),
            pytest.param(["--tvdb-id", "7"], [(Arr.SONARR, 7)], id="tvdb-id"),
        ],
    )
    def test_the_flags_select_the_arrs(
        self,
        recorder: _RunArrsRecorder,
        flags: list[str],
        arrs: list[tuple[Arr, int | None]],
    ) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single", *flags])
        assert result.exit_code == 0
        assert recorder.arrs == arrs

    def test_the_wait_mode_flag_reaches_the_run(self, recorder: _RunArrsRecorder) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single", "--import-wait-mode", "off"])
        assert result.exit_code == 0
        assert recorder.import_wait_mode is ImportWaitMode.OFF

    @pytest.mark.parametrize("flag", ["--movie-id", "--series-id"])
    def test_the_old_id_flags_are_usage_errors(self, recorder: _RunArrsRecorder, flag: str) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single", flag, "1"])
        assert result.exit_code == 2
        assert recorder.arrs is None

    def test_hybrid_is_not_a_wait_mode(self, recorder: _RunArrsRecorder) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single", "--import-wait-mode", "hybrid"])
        assert result.exit_code == 2
        assert recorder.arrs is None


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

    def test_implicit_selection_skips_the_unconfigured_arr(self, recording: RecordingHub) -> None:
        kept = configured_arrs(
            [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR)],
            _config_with(sonarr=True),
            explicit=False,
            config_path="config.yml",
        )
        assert kept == [(Arr.SONARR, None)]
        # The note is a first-party INFO Diagnostic on the hub: a Sonarr-only
        # setup is normal, not an error. The message is FLAT (the live console
        # indents it by its position inside the open boot scope). The arr name
        # is capitalized prose, not a lowercase config key.
        (skip,) = recording.of_type(Diagnostic)
        assert skip.severity is Severity.INFO
        assert skip.message == "Radarr not configured - not run"
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
        assert warning.message == "radarr.url is set but radarr.api_key is not - Radarr is not run"

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

    def test_explicit_selection_of_an_unconfigured_arr_refuses(self) -> None:
        # A Sonarr-only setup asked for Radarr by flag: a config mistake, so
        # nothing runs, not even the configured Sonarr.
        config = _write_current_config("sonarr:\n  url: http://127.0.0.1:9\n  api_key: k\n")

        result = _invoke_offline("run", "single", "--radarr")

        assert result.exit_code == 1
        assert _lines(result.stdout)[-1] == (
            f"ERROR [Pearlarr] Radarr was selected but is not configured - set radarr.url and radarr.api_key "
            f"in {config} during=boot"
        )

    def test_nothing_configured_reports_and_refuses(self) -> None:
        # The refusal comes before the mapping sources are fetched: a run with
        # nothing to do never downloads them (`_invoke_offline` fails on any request).
        config = _write_current_config()

        result = _invoke_offline("run", "single")

        assert result.exit_code == 1
        assert _lines(result.stdout)[-3:] == [*_SKIP_NOTES, _nothing_configured_line(config)]

    def test_fully_configured_passes_through_unchanged(self, recording: RecordingHub) -> None:
        arrs: list[ArrTarget] = [ArrTarget(Arr.RADARR), ArrTarget(Arr.SONARR, 7)]
        kept = configured_arrs(arrs, _config_with(sonarr=True, radarr=True), explicit=True, config_path="config.yml")
        assert kept == arrs
        assert recording.of_type(Diagnostic) == []


# Mapping stanza shared by the runnable-config helpers: disabled sources keep
# the resolver off the network, so a run gets past the shared-deps stage and
# into the per-arr loop without any I/O.
_NO_MAPPINGS = "mappings:\n  anime_ids: false\n  anime_lists: false\n  anibridge: false\n"


def _write_runnable_config() -> None:
    """A config with Sonarr configured and every mapping source disabled."""

    _write_data_dir_config("sonarr:\n  url: http://sonarr:8989\n  api_key: k\n" + _NO_MAPPINGS)


def _write_config_fetching_anime_ids() -> None:
    """A current config with Sonarr configured and the anime ids file as the one mapping source on."""

    _write_current_config(
        "sonarr:\n  url: http://127.0.0.1:9\n  api_key: k\nmappings:\n  anime_lists: false\n  anibridge: false\n"
    )


class TestRunFailuresAreCleanAndNonzero:
    """Every failed arr run returns False (exit 1) with a one-liner, not a traceback."""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            pytest.param("qbit", "qBittorrent connection failed", id="qbit-connection"),
            pytest.param(
                "unreachable",
                # The error's message carries the address that was tried.
                "Sonarr run failed - Could not reach Sonarr at http://sonarr:8989 "
                "(request failed (ConnectError)) - check sonarr.url",
                id="arr-unreachable",
            ),
            pytest.param("unauthorized", "Sonarr rejected the API key - check sonarr.api_key", id="arr-unauthorized"),
            pytest.param(
                "contract",
                # The library answered but validated to nothing: the one-line
                # contract handler, never the traceback handler.
                "Sonarr run failed - none of the 3 SonarrSeries records validated",
                id="arr-contract",
            ),
        ],
    )
    def test_a_failed_arr_run_reports_cleanly_and_returns_false(
        self,
        error: str,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pearlarr.arr_http import ArrAuthError, ArrUnreachableError
        from pearlarr.arr_run_services import ArrRunDeps, QbitConnectionError
        from pearlarr.boundary import BoundaryContractError

        exceptions: dict[str, Exception] = {
            "qbit": QbitConnectionError("qBittorrent connection failed - check the host and credentials"),
            "unreachable": ArrUnreachableError(
                "Could not reach Sonarr at http://sonarr:8989 (request failed (ConnectError))",
            ),
            "unauthorized": ArrAuthError("Sonarr at http://sonarr:8989 rejected the API key (status code 401)"),
            "contract": BoundaryContractError(
                "none of the 3 SonarrSeries records validated. Refusing to treat it as empty",
            ),
        }

        def fail_build(*args: object, **kwargs: object) -> NoReturn:
            raise exceptions[error]

        monkeypatch.setattr(ArrRunDeps, "build", fail_build)
        _write_runnable_config()

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert expected in out
        assert "Traceback" not in out

    def test_malformed_yaml_fails_the_run_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The syntax error sits ON the credential line: the report must carry the
        # problem + position (from the error's parts), never str(e)'s source
        # snippet, which would quote the api key back to the console/log.
        _write_data_dir_config('sonarr:\n  api_key: "hunter2\n')

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert "Unreadable YAML" in out
        assert "line 3" in out  # the position survives, so it is easy to find
        assert "hunter2" not in out
        assert "Traceback" not in out

    def test_a_mapping_source_failure_fails_the_run(self) -> None:
        # A cached anime ids file that won't parse: still fresh, so nothing is
        # downloaded, and reading it fails.
        _write_config_fetching_anime_ids()
        Path(resolve_paths().data_dir, "anime_ids.json").write_text("{not json", encoding="utf-8")

        result = _invoke_offline("run", "single", "--sonarr")

        assert result.exit_code == 1
        assert "ERROR [Pearlarr] Could not fetch/parse the mapping sources - skipping this run during=boot" in (
            _lines(result.stdout)
        )

    def test_a_mapping_download_network_failure_reports_cleanly(self) -> None:
        # A brand-new user with no network hits the first-ever source download,
        # which has no cached copy to fall back on. That must surface as one line
        # with a hint, not a traceback.
        _write_config_fetching_anime_ids()

        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=httpx.ConnectError("no network"))
            result = CliRunner().invoke(pearlarr_cli, ["run", "single", "--sonarr"])

        assert result.exit_code == 1
        assert _lines(result.stdout)[-1] == (
            "ERROR [Pearlarr] Could not download the mapping sources (download failed: ConnectError) "
            "- check your network connection - skipping this run during=boot"
        )

    def test_an_invalid_config_fails_the_run_listing_the_keys(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_data_dir_config("sonar:\n  url: x\n")

        assert run_single(sonarr=True) is False
        out = capsys.readouterr().out
        assert "Invalid configuration" in out
        assert "sonar" in out


class TestConfigInspection:
    """`config validate` / `config show`: report cleanly, never write files."""

    def test_validate_missing_file_points_at_init_and_writes_nothing(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert config_validate() is False
        assert "config init" in capsys.readouterr().err
        assert not os.path.exists(resolve_paths().config_file)

    def test_validate_lists_the_bad_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_data_dir_config("sonar:\n  url: x\n")
        assert config_validate() is False
        err = capsys.readouterr().err
        assert "Invalid configuration" in err
        assert "  - sonar: unknown key" in err

    def test_validate_reports_malformed_yaml_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_data_dir_config('sonarr:\n  api_key: "hunter2\n')
        assert config_validate() is False
        captured = capsys.readouterr()
        assert "Unreadable YAML" in captured.err
        # Never str(e): its snippet quotes the offending (credential) line.
        assert "hunter2" not in captured.err + captured.out

    def test_validate_reports_what_a_run_would_use(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_data_dir_config("sonarr:\n  url: http://sonarr:8989\n  api_key: k\n")
        assert config_validate() is True
        out = capsys.readouterr().out
        assert "OK" in out
        assert "sonarr:      configured" in out
        assert "radarr:      not configured" in out
        assert "dry runs only" in out  # no qbittorrent credentials -> nothing is grabbed

    def test_show_masks_set_secrets_and_keeps_unset_ones_null(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_data_dir_config(
            "sonarr:\n  url: http://sonarr:8989\n  api_key: hunter2\n"
            "notifications:\n  discord_url: https://discord.com/api/webhooks/1/tok\n"
        )
        assert config_show() is True
        out = capsys.readouterr().out
        assert "hunter2" not in out
        assert "tok" not in out
        assert "api_key: REDACTED" in out
        assert "discord_url: REDACTED" in out
        # The unset radarr key stays null: "is it even set?" must stay answerable.
        assert "api_key: null" in out

    def test_validate_reports_an_old_schema_with_its_conversions(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Valid-but-old is a status fact, not a failure: the run migrates it in
        # memory, and validate names the command that updates the file itself.
        _write_data_dir_config("seadex:\n  private_releases: allow\n")
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
        _write_data_dir_config("radarr:\n  url: http://radarr:7878\n")
        assert config_validate() is True  # the file itself is valid
        assert "radarr.api_key is not set" in capsys.readouterr().out

    def test_show_masks_the_options_block_and_url_logins(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_data_dir_config(
            "sonarr:\n  url: http://user:basicpass@sonarr:8989\n  api_key: k\n"
            "qbittorrent:\n"
            "  host: http://qbit:8080\n"
            "  username: qbcanaryuser\n"
            "  password: hunter2\n"
            "  options:\n    REQUESTS_ARGS:\n      headers:\n        Authorization: Bearer sekrit\n"
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
        text = "seadex:\n  public_only: true\nsonarr:\n  url: http://s\n  api_key: hunter2\n"
        config = _write_data_dir_config(text)

        assert config_migrate() is True

        out = capsys.readouterr().out
        assert "previous file saved as" in out
        assert "public_only" in out
        # The command echoes paths and notes, never config values.
        assert "hunter2" not in out
        assert Path(f"{config}.bak").read_text(encoding="utf-8") == text
        migrated = config.read_text(encoding="utf-8")
        assert "api_key: hunter2" in migrated
        assert "public_only" not in migrated
        assert f"config_version: {CONFIG_VERSION}" in migrated

    def test_a_current_file_is_nothing_to_do(self, capsys: pytest.CaptureFixture[str]) -> None:
        text = f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://s\n"
        config = _write_data_dir_config(text)

        assert config_migrate() is True

        assert "nothing to do" in capsys.readouterr().out
        assert config.read_text(encoding="utf-8") == text
        assert not os.path.exists(f"{config}.bak")

    def test_missing_file_points_at_init_and_writes_nothing(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert config_migrate() is False
        assert "config init" in capsys.readouterr().err
        assert not os.path.exists(resolve_paths().config_file)

    def test_an_invalid_file_reports_and_stays_untouched(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        text = "seadex:\n  public_only: true\ntypo_key: 1\n"
        config = _write_data_dir_config(text)

        assert config_migrate() is False

        err = capsys.readouterr().err
        assert "Invalid configuration" in err
        assert "typo_key" in err
        assert config.read_text(encoding="utf-8") == text
        assert not os.path.exists(f"{config}.bak")

    def test_refuses_while_a_run_holds_the_lock(self, capsys: pytest.CaptureFixture[str]) -> None:
        # A run rewrites an old config itself, under the same lock: the two
        # writers must never meet on the backup and temp files.
        text = "seadex:\n  public_only: true\n"
        config = _write_data_dir_config(text)
        data_dir = resolve_paths().data_dir

        with single_instance_lock(data_dir):
            assert config_migrate() is False

        assert capsys.readouterr().err == f"Another Pearlarr run is active in {data_dir} - not modifying the config\n"
        assert config.read_text(encoding="utf-8") == text
        assert not os.path.exists(f"{config}.bak")


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


class TestLogLevelWiring:
    """`advanced.log_level` applies once the config is read, and `--log-level` wins over it.

    The skip notes are INFO lines that come after the read, so a config at ERROR hides them.
    """

    def test_the_config_level_is_applied_to_a_run(self) -> None:
        config = _write_current_config("advanced:\n  log_level: ERROR\n")

        result = _invoke_offline("run", "single")

        lines = _lines(result.stdout)
        assert [line for line in lines if line in _SKIP_NOTES] == []
        assert lines[-1] == _nothing_configured_line(config)

    def test_the_cli_override_wins_over_the_config(self) -> None:
        _write_current_config("advanced:\n  log_level: ERROR\n")

        result = _invoke_offline("run", "single", "--log-level", "info")

        assert [line for line in _lines(result.stdout) if line in _SKIP_NOTES] == _SKIP_NOTES

    def test_the_cli_override_wins_in_scheduled_mode_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_current_config("advanced:\n  log_level: ERROR\n")

        result, _ = _run_scheduled_once(monkeypatch, "--log-level", "info")

        assert [line for line in _lines(result.stdout) if line in _SKIP_NOTES] == _SKIP_NOTES

    def test_the_cli_level_holds_from_the_first_line(self) -> None:
        # The flag is known before the config is read, so the opening lines obey it too.
        config = _write_current_config()

        result = _invoke_offline("run", "single", "--log-level", "warning")

        assert _lines(result.stdout) == [_nothing_configured_line(config)]


class _StopScheduledLoop(Exception):
    """Ends `run scheduled`'s endless loop from its faked sleep."""


def _run_scheduled_once(
    monkeypatch: pytest.MonkeyPatch, *args: str, schedule_time: str | None = None
) -> tuple[Result, list[float]]:
    """`pearlarr run scheduled` through its first run, offline, and the seconds it then went to sleep for.

    `schedule_time` is the SCHEDULE_TIME the loop sees (None leaves it unset). The faked sleep ends the loop.
    """

    if schedule_time is None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
    else:
        monkeypatch.setenv("SCHEDULE_TIME", schedule_time)
    slept: list[float] = []

    def sleep(seconds: float) -> NoReturn:
        slept.append(seconds)
        raise _StopScheduledLoop

    monkeypatch.setattr(time, "sleep", sleep)
    result = _invoke_offline("run", "scheduled", *args)
    assert isinstance(result.exception, _StopScheduledLoop)
    return result, slept


class _SignalRecorder:
    """A fake `signal.signal` that records each registration."""

    def __init__(self) -> None:
        self.registered: list[tuple[int, object]] = []

    def __call__(self, signum: int, handler: object) -> None:
        self.registered.append((signum, handler))


class TestLogFormatWiring:
    """`advanced.log_format` is what both run commands print in, terminal or not.

    json is one JSON object per line. rich is the indented console layout.
    """

    @staticmethod
    def _events(stdout: str) -> list[dict[str, object]]:
        # A line that isn't JSON fails the test here.
        return [cast("dict[str, object]", json.loads(line)) for line in stdout.splitlines()]

    def test_run_single_threads_the_config_format(self) -> None:
        config = _write_current_config("advanced:\n  log_format: json\n")

        result = _invoke_offline("run", "single")

        events = self._events(result.stdout)
        assert events[0]["event"] == "pearlarr_started"
        assert [event["message"] for event in events if event["event"] == "diagnostic"] == [
            "Radarr not configured - not run",
            "Sonarr not configured - not run",
            (
                "Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, "
                f"or radarr.url and radarr.api_key, in {config}"
            ),
        ]

    def test_run_scheduled_threads_the_config_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_current_config("advanced:\n  log_format: json\n")

        result, _ = _run_scheduled_once(monkeypatch)

        events = self._events(result.stdout)
        assert (events[0]["event"], events[-1]["event"]) == ("scheduled_run_started", "next_run_scheduled")

    def test_a_rich_config_prints_the_rich_layout_off_a_terminal(self) -> None:
        # The runner's stdout is no terminal, so "auto" would have printed plain lines.
        _write_current_config("advanced:\n  log_format: rich\n")

        result = _invoke_offline("run", "single")

        assert "  Sonarr not configured - not run" in result.stdout.splitlines()

    def test_a_rich_config_prints_the_rich_layout_in_scheduled_mode_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_current_config("advanced:\n  log_format: rich\n")

        result, _ = _run_scheduled_once(monkeypatch)

        lines = result.stdout.splitlines()
        assert "  Sonarr not configured - not run" in lines
        assert lines[-1].startswith("Next scheduled run at ")


class TestMissingConfigExitsNonzero:
    """A virgin data dir: exit 1 with the template written, never a silent retry.

    Driven end-to-end through CliRunner: the `typer.Exit` raised inside
    `load_shared_config`'s FileNotFoundError handler must escape `run_arrs`'
    finallys (boot region, web client, run lock all release) and reach typer as
    exit code 1 - pinning the whole propagation path, not just the helper.
    """

    def test_run_single_writes_the_template_and_exits_one(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1

        config = Path(resolve_paths().config_file)
        assert config.read_text(encoding="utf-8") == _starter_text()
        assert "a template was written" in result.output
        # This handler exits now. The skip-and-retry wording belongs to the others.
        assert "Skipping this run" not in result.output

    def test_bare_pearlarr_runs_scheduled_mode_and_exits_one(self) -> None:
        # No subcommand is the scheduled loop. A missing config stops it at once
        # instead of sleeping on a template nobody has filled in.
        result = _invoke_offline()

        assert result.exit_code == 1
        assert _lines(result.stdout)[0] == "INFO [run] scheduled run started number=1"
        assert Path(resolve_paths().config_file).read_text(encoding="utf-8") == _starter_text()


class TestOutdatedEnvNamesWarning:
    """A version 6 override variable name warns once per load, every run."""

    def test_old_names_warn_in_one_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        recording = install_recording_hub()
        monkeypatch.setenv("PEARLARR_IMPORTS__WAIT_TIMEOUT", "7")
        monkeypatch.setenv("PEARLARR_ADVANCED__CACHE_TIME", "3")
        config = tmp_path / "config.yml"
        config.write_text(f"config_version: {CONFIG_VERSION}\n", encoding="utf-8")
        config.chmod(0o600)

        assert load_shared_config(str(config), BootFlow(), "") is not None

        (warning,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert warning.message == (
            "Outdated environment variables (they stop working in 1.7.0): "
            "PEARLARR_ADVANCED__CACHE_TIME is now PEARLARR_MAPPINGS__REFRESH_DAYS; "
            "PEARLARR_IMPORTS__WAIT_TIMEOUT is now PEARLARR_IMPORTS__DOWNLOAD_TIMEOUT"
        )


class TestUnknownTrackerWarning:
    """Unknown `seadex.trackers` values warn at load - they silently match nothing.

    Warn-not-reject: strict key validation already rejects unknown KEYS. An unknown
    VALUE is a typo that would quietly filter out every release from that tracker.
    """

    @staticmethod
    def _load(tmp_path: Path, body: str) -> AppConfig | None:
        config = tmp_path / "config.yml"
        # Current-schema version + 0600: keep the old-schema and loose-permissions
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
    """`load_shared_config`'s last-resort handler: ERROR with traceback, then skip."""

    def test_unexpected_load_error_reports_and_skips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No file content reaches this handler (undecodable bytes are a YAMLError,
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

# The canonical v0-schema config body: old enough to trigger auto-migrate,
# used across TestConfigAutoMigrate and TestScheduledLifecycle.
_OLD_SCHEMA_CONFIG = "seadex:\n  private_releases: allow\nsonarr:\n  url: http://s\n  api_key: k\n"
# The same with no arr, so a run through the CLI stops at the selection, before any request.
_OLD_SCHEMA_NO_ARRS = "seadex:\n  private_releases: allow\n"


class TestConfigAutoMigrate:
    """An old-schema config is rewritten in place after the load, backup first.

    The console gets one terse line either way. The functional detail (what was
    converted) goes to the log file only. A file that cannot be
    rewritten falls back to the in-memory migration behind a short warning, and
    a dry run never writes at all.
    """

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        config = tmp_path / "config.yml"
        config.write_text(body, encoding="utf-8")
        config.chmod(0o600)  # keep the loose-permissions warn out of the stream
        return config

    @staticmethod
    def _load(config: Path) -> AppConfig:
        loaded = load_shared_config(str(config), BootFlow(), "")
        assert loaded is not None
        return loaded

    def test_an_old_schema_file_is_rewritten_with_a_backup(self) -> None:
        config = _write_data_dir_config(_OLD_SCHEMA_NO_ARRS)
        config.chmod(0o600)  # keep the loose-permissions warn out of the stream

        result = _invoke_offline("run", "single")

        rewritten = config.read_text(encoding="utf-8")
        assert f"config_version: {CONFIG_VERSION}" in rewritten
        assert "private_releases: warn" in rewritten
        assert Path(f"{config}.bak").read_text(encoding="utf-8") == _OLD_SCHEMA_NO_ARRS
        # After the opening line and the config read: one line about the
        # rewrite, and no warning about the old schema.
        assert _lines(result.stdout)[2:] == [
            (
                "INFO [Pearlarr] Config migrated to the current schema - previous file saved as "
                f"{config}.bak (details in the log) during=boot"
            ),
            *_SKIP_NOTES,
            _nothing_configured_line(config),
        ]

    def test_the_conversion_detail_goes_to_the_file_log(self) -> None:
        _write_data_dir_config(_OLD_SCHEMA_NO_ARRS).chmod(0o600)

        result = _invoke_offline("run", "single")

        log_lines = _lines(Path(resolve_paths().log_dir, "Pearlarr.log").read_text(encoding="utf-8"))
        (detail,) = [
            line for line in log_lines if line.startswith(f"INFO [Pearlarr] Config schema v0 -> v{CONFIG_VERSION}: ")
        ]
        assert "'allow'" in detail  # the private_releases conversion
        assert "wait_mode" in detail  # the note-only default flip comes along
        # The console prints every other line of the log.
        assert _lines(result.stdout) == [line for line in log_lines if line != detail]

    def test_a_raised_log_level_drops_the_details_promise(self, tmp_path: Path) -> None:
        # At --log-level warning the INFO detail line never reaches the file
        # log, so the console note must not point at it.
        recording = install_recording_hub()
        current_hub().set_level(int(Severity.WARNING))
        config = self._write(tmp_path, _OLD_SCHEMA_CONFIG)

        _rewrite_old_config(str(config), self._load(config), dry_run=False)

        (notice,) = [d for d in recording.of_type(Diagnostic) if not d.file_only and d.severity is Severity.INFO]
        assert notice.message.endswith(".bak")
        assert "details in the log" not in notice.message

    def test_a_dry_run_leaves_the_file_alone_behind_a_note(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        config = self._write(tmp_path, _OLD_SCHEMA_CONFIG)

        _rewrite_old_config(str(config), self._load(config), dry_run=True)

        assert config.read_text(encoding="utf-8") == _OLD_SCHEMA_CONFIG
        assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yml"]
        (note,) = [d for d in recording.of_type(Diagnostic) if not d.file_only]
        assert note.severity is Severity.INFO
        assert note.message == "Config uses an older schema - left unchanged on a dry run"

    @_needs_posix_permissions
    def test_an_unwritable_file_falls_back_in_memory_with_a_short_warning(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        config = self._write(tmp_path, _OLD_SCHEMA_CONFIG)
        loaded = self._load(config)
        tmp_path.chmod(0o500)  # the temp-file create fails before anything is touched
        try:
            _rewrite_old_config(str(config), loaded, dry_run=False)
        finally:
            tmp_path.chmod(0o755)

        assert config.read_text(encoding="utf-8") == _OLD_SCHEMA_CONFIG
        assert not (tmp_path / "config.yml.bak").exists()
        (warning,) = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING and not d.file_only]
        assert warning.message == (
            f"Config {config} uses an older schema and could not be rewritten in place - "
            f"run pearlarr config migrate to update the file (a backup is kept)"
        )
        reason = next(d for d in recording.of_type(Diagnostic) if d.file_only and d.severity is Severity.WARNING)
        assert str(config) in reason.message

    def test_a_current_schema_file_is_left_alone(self, tmp_path: Path) -> None:
        recording = install_recording_hub()
        body = f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://s\n  api_key: k\n"
        config = self._write(tmp_path, body)

        _rewrite_old_config(str(config), self._load(config), dry_run=False)

        assert config.read_text(encoding="utf-8") == body
        assert not (tmp_path / "config.yml.bak").exists()
        assert recording.of_type(Diagnostic) == []


class TestLooseConfigPermissions:
    """A config other users can read warns with the command that fixes it."""

    @_needs_posix_permissions
    def test_a_world_readable_config_warns_with_the_fix(self) -> None:
        config = _write_current_config()
        config.chmod(0o644)

        result = _invoke_offline("run", "single")

        assert [line for line in _lines(result.stdout) if line.startswith("WARNING")] == [
            (
                f"WARNING [Pearlarr] Config file {config} is readable by other users and holds API keys - "
                f"tighten it with: chmod 600 {config} during=boot"
            )
        ]


class TestUnwritableDataDir:
    """An unwritable data directory: one actionable stderr line + exit 1, no traceback.

    `ensure_data_dir`/`setup_logger` run before any logger exists, so the
    report must go straight to stderr. `config init`'s template copy and
    `cache restore`'s copy get the same treatment. An unreadable CONFIG in a
    healthy dir instead keeps the skip+retry contract - exiting would kill the
    scheduled daemon.
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
        # run. The install-time probe must abort cleanly, not strike the renderer.
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
    def test_cache_restore_into_a_readonly_dir_leaves_the_db_alone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The temp copy beside cache.db is what fails: one line, and the live db as it was.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(data_dir))
        write_cache_db(data_dir)
        assert CliRunner().invoke(pearlarr_cli, ["cache", "backup"]).exit_code == 0
        live = (data_dir / "cache.db").read_bytes()
        data_dir.chmod(0o500)
        try:
            result = CliRunner().invoke(pearlarr_cli, ["cache", "restore"])
        finally:
            data_dir.chmod(0o755)

        assert result.exit_code == 1
        assert result.stderr.startswith("Cache restore failed (")
        assert result.stderr.endswith(") - cache.db is left unchanged\n")
        assert (data_dir / "cache.db").read_bytes() == live

    @_needs_posix_permissions
    def test_an_unreadable_config_skips_the_run_instead_of_exiting(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Handler order in load_shared_config matters twice: FileNotFoundError
        # (exit 1) must not swallow this OSError, and the OSError handler must
        # catch it before the traceback handler. Skip + retry, never exit.
        config = _write_data_dir_config("sonarr:\n  url: http://s\n  api_key: k\n")
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
        # pearlarr_started line (plain renderer, structured grammar) carries the
        # data_dir fact FIRST, before the config read fails.
        result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1
        assert "Pearlarr started" in result.output
        assert f"data_dir={resolve_paths().data_dir}" in result.output


class TestScheduledLifecycle:
    """The run commands' lifecycle: config handling, the SIGTERM handler both install, run events, the help text."""

    @staticmethod
    def _stub_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stop the run right after the config load (no network)."""

        def refuse_resolver(*args: object, **kwargs: object) -> None:
            return None

        monkeypatch.setattr("pearlarr.bootstrap.build_resolver", refuse_resolver)

    def test_invalid_config_still_skips_and_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only the missing file exits: an invalid config is likely mid-edit, so
        # the loop reports it, skips the run and sleeps until the next one.
        config = _write_current_config("sonar:\n  url: x\n")

        result, slept = _run_scheduled_once(monkeypatch)

        assert [line for line in _lines(result.stdout) if line.startswith("ERROR")] == [
            (
                f"ERROR [Pearlarr] Invalid configuration in {config}:\\n  - sonar: unknown key\\n"
                "Fix the listed keys and re-run - skipping this run - will retry in 6h (Ctrl-C to stop) during=boot"
            )
        ]
        assert slept == [6 * 3600]

    def test_a_dry_run_never_rewrites_the_config_file(
        self,
        logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Live repro of the original bug: --dry-run rewrote an 81 B old-schema
        # file to the full 12 kB template. The file must stay byte-identical.
        recording = install_recording_hub()
        old = _OLD_SCHEMA_CONFIG
        _write_data_dir_config(old)
        paths = resolve_paths()
        self._stub_resolver(monkeypatch)

        run_arrs(
            [ArrTarget(Arr.SONARR)],
            paths=paths,
            logger=logger,
            file_log_renderer=FileLogRenderer(paths.log_dir),
            dry_run=True,
        )

        assert Path(paths.config_file).read_text(encoding="utf-8") == old
        assert not os.path.exists(paths.config_file + ".bak")
        assert any("left unchanged on a dry run" in d.message for d in recording.of_type(Diagnostic))

    def test_a_run_deletes_log_backups_older_than_the_retention(self) -> None:
        _write_current_config("advanced:\n  log_retention_days: 3\n")
        logs = Path(resolve_paths().log_dir)
        logs.mkdir()
        now = time.time()
        for name, days_old in [("Pearlarr.log.five-days-old", 5), ("Pearlarr.log.one-day-old", 1)]:
            backup = logs / name
            backup.write_text("an earlier run\n", encoding="utf-8")
            os.utime(backup, (now - days_old * 86400, now - days_old * 86400))

        _invoke_offline("run", "single")

        assert sorted(path.name for path in logs.glob("Pearlarr.log.*")) == ["Pearlarr.log.one-day-old"]

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

    def test_sigterm_handler_ignores_a_repeat_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _SignalRecorder()
        monkeypatch.setattr(signal, "signal", recorder)

        with pytest.raises(SystemExit):
            _handle_sigterm(signal.SIGTERM, None)

        assert recorder.registered == [(signal.SIGTERM, signal.SIG_IGN)]

    def test_a_scheduled_run_states_its_number_and_the_next_run_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The number opens the console's lines, so it came after the run began
        # and sits in this run's log file, not in the last run's.
        _write_current_config()
        before = datetime.now().astimezone()

        result, _ = _run_scheduled_once(monkeypatch)

        lines = _lines(result.stdout)
        assert lines[0] == "INFO [run] scheduled run started number=1"
        prefix = "INFO [run] next run scheduled at="
        assert lines[-1].startswith(prefix)
        next_run = datetime.fromisoformat(lines[-1].removeprefix(prefix))
        assert next_run.tzinfo is not None  # the time carries its UTC offset
        assert abs((next_run - (before + timedelta(hours=6))).total_seconds()) < 60

    def test_a_scheduled_run_asks_for_every_arr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_current_config()

        result, _ = _run_scheduled_once(monkeypatch)

        assert [line for line in _lines(result.stdout) if "not configured - not run" in line] == _SKIP_NOTES

    def test_run_scheduled_registers_the_sigterm_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _SignalRecorder()
        monkeypatch.setattr(signal, "signal", recorder)

        _invoke_offline("run", "scheduled")  # a fresh data dir: the run stops at the missing config

        assert recorder.registered == [(signal.SIGTERM, _handle_sigterm)]

    def test_run_single_registers_the_sigterm_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _SignalRecorder()
        monkeypatch.setattr(signal, "signal", recorder)
        during_run: list[tuple[int, object]] = []

        def run_arrs(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            during_run.extend(recorder.registered)
            return True

        monkeypatch.setattr("pearlarr.bootstrap.run_arrs", run_arrs)

        assert run_single() is True
        assert during_run == [(signal.SIGTERM, _handle_sigterm)]

    def test_scheduled_help_names_the_bare_metal_fallback(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["run", "scheduled", "--help"])
        assert result.exit_code == 0
        assert "bare-metal" in result.output


class TestRunLockContention:
    """A second run against the same data dir warns once and exits 1."""

    def test_a_run_skips_while_another_holds_the_lock(self) -> None:
        # It stops before the config read: none exists here, and none is written.
        data_dir = resolve_paths().data_dir
        os.makedirs(data_dir)

        with single_instance_lock(data_dir) as held:
            assert held
            result = _invoke_offline("run", "single")

        assert result.exit_code == 1
        assert _lines(result.stdout) == [
            f"WARNING [Pearlarr] Another Pearlarr run is active in {data_dir} - skipping this run"
        ]
        assert not os.path.exists(resolve_paths().config_file)


_INTERVAL_2_5 = "schedule:\n  interval_hours: 2.5\n"
_INTERVAL_INVALID = "schedule:\n  interval_hours: -1\n"  # fails validation


class TestScheduleHours:
    """How long the loop sleeps: a valid SCHEDULE_TIME (deprecated) > schedule.interval_hours > 6.

    A bad SCHEDULE_TIME is named along with the hours actually used. A config
    that won't load falls back to 6 without a warning of its own (the load reports it).
    """

    @pytest.mark.parametrize(
        ("schedule_time", "body", "hours", "warning"),
        [
            pytest.param(None, "", 6, None, id="default"),
            pytest.param(None, _INTERVAL_2_5, 2.5, None, id="config"),
            pytest.param(
                "0.5",
                _INTERVAL_2_5,
                0.5,
                "SCHEDULE_TIME is deprecated - set schedule.interval_hours in the config instead",
                id="env-wins",
            ),
            *(
                pytest.param(
                    raw, _INTERVAL_2_5, 2.5, f"Invalid SCHEDULE_TIME {raw!r} - using 2.5 hours", id=f"bad-env-{raw}"
                )
                for raw in ["banana", "0", "-3", "inf", "nan"]
            ),
            pytest.param(
                "banana",
                _INTERVAL_INVALID,
                6,
                "Invalid SCHEDULE_TIME 'banana' - using 6 hours",
                id="bad-env-bad-config",
            ),
            pytest.param(None, _INTERVAL_INVALID, 6, None, id="invalid-config"),
            pytest.param(None, "schedule: [unclosed\n", 6, None, id="malformed-config"),
        ],
    )
    def test_the_loop_sleeps_for_the_interval_that_wins(
        self,
        schedule_time: str | None,
        body: str,
        hours: float,
        warning: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_current_config(body)

        result, slept = _run_scheduled_once(monkeypatch, schedule_time=schedule_time)

        assert slept == [hours * 3600]
        warnings = [line for line in _lines(result.stdout) if line.startswith("WARNING")]
        assert warnings == ([] if warning is None else [f"WARNING [Pearlarr] {warning}"])


def test_commands_verify_tls_against_the_os_trust_store() -> None:
    """The root callback swaps `ssl.SSLContext` for truststore's OS-backed one, before any HTTP client builds a context."""

    truststore.extract_from_ssl()  # an earlier command in this process already swapped it in
    try:
        assert CliRunner().invoke(pearlarr_cli, ["paths"]).exit_code == 0
        assert ssl.SSLContext is truststore.SSLContext
    finally:
        truststore.extract_from_ssl()


def test_cache_stats_pins_the_padded_human_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The renderer reproduces the pre-hub typer.echo block byte-for-byte:
    # 17-column padded labels and a two-decimal MiB size.
    monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
    write_cache_db(tmp_path)

    result = CliRunner().invoke(pearlarr_cli, ["cache", "stats"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[:7] == [
        "entries:         1",
        "anilist_meta:    0",
        "sonarr_parse:    0",
        "pending_imports: 0",
        "guard_facts:     0",
        "tvdb_orders:     0",
        "scene_mappings:  0",
    ]
    assert re.fullmatch(r"size:            \d+\.\d\d MiB", lines[7])
    assert len(lines) == 8
