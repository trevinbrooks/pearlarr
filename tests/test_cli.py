# pyright: strict
# pyright: reportPrivateUsage=false
# ``_schedule_hours`` is a deliberately under-test private helper; the repo already
# disables reportPrivateUsage for all of tests/, but the strict directive above
# re-enables it, so restore the repo's test policy here.
"""Tests for the SeaDexArr CLI commands.

Pins the behaviours the commands must guarantee:

* A corrupt / not-a-database ``cache.db`` is *reported*, never crashed on - the
  ``stats`` and ``check`` diagnostics (and ``backup``) return a clean False instead
  of letting a ``sqlite3`` traceback escape (finding #4). ``check`` exists to report
  bad integrity, so it must survive the very corruption it diagnoses.
* The destructive commands (``restore`` / ``remove``) take the single-instance run
  lock first and refuse while a run is active, so they never unlink or replace the
  live db out from under it (finding #5).
* Failure paths report cleanly (missing files echo one line, not a traceback) and
  a False return maps to exit code 1 through the Typer apps' result callback.
* ``config init`` never overwrites a filled-in config.yml without ``--force``.

Each test points ``resolve_paths()`` at its own ``tmp_path`` via ``SEADEX_ARR_DATA_DIR``
and calls the command functions directly (they return ``bool``); the exit-code
tests go through ``CliRunner`` since the callback only runs inside typer.
"""

from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from seadexarr.modules.cache import CacheStore
from seadexarr.modules.cli import (
    _schedule_hours,
    cache_backup,
    cache_check,
    cache_remove,
    cache_restore,
    cache_stats,
    config_init,
    run_single,
    seadexarr_cli,
)
from seadexarr.modules.config import Arr, template_path
from seadexarr.modules.runlock import single_instance_lock


def _build_cache(tmp_path: Path) -> None:
    """Write a real on-disk ``cache.db`` under ``tmp_path`` holding one entry.

    Uses the normal load -> stage -> ``save(preview=False)`` path, which promotes
    the in-memory db to the file, so the fixture matches a cache a real run leaves
    behind (entry ``(SONARR, 7)`` with ``name="X"``).
    """

    store = CacheStore.load(str(tmp_path / "cache.db"), config_checksum="x")
    store.update_cache(Arr.SONARR, 7, {"name": "X"})
    store.save(preview=False)
    store.close()


class TestCacheRoundTrip:
    def test_backup_then_restore_preserves_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_backup() is True
        assert (tmp_path / "cache.backup.db").exists()

        # A stale WAL left next to cache.db must not shadow the restored snapshot;
        # restore clears the sidecars before swapping the copy into place.
        (tmp_path / "cache.db-wal").write_text("stale")

        assert cache_restore() is True
        # Copy-restore: the backup SURVIVES (a post-restore corruption can be
        # restored again) and no temp file is left behind.
        assert (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.db.tmp").exists()
        assert not (tmp_path / "cache.db-wal").exists()  # stale sidecar cleared

        # Restore is repeatable off the surviving backup.
        assert cache_restore() is True

        # The restored db still holds the original entry.
        store = CacheStore.open_readonly(str(tmp_path / "cache.db"))
        try:
            entry = store.get_entry(Arr.SONARR, 7)
            assert entry is not None
            assert entry.name == "X"
        finally:
            store.close()

    def test_remove_deletes_db_and_sidecars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        (tmp_path / "cache.db-wal").write_text("stale")
        (tmp_path / "cache.db-shm").write_text("stale")

        assert cache_remove() is True
        assert not (tmp_path / "cache.db").exists()
        assert not (tmp_path / "cache.db-wal").exists()
        assert not (tmp_path / "cache.db-shm").exists()


class TestHealthyDiagnostics:
    def test_stats_and_check_report_a_healthy_db(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
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
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # Reporting bad integrity is this command's whole job, so it must not crash
        # on the very corruption it diagnoses.
        assert cache_check() is False
        assert "integrity" in capsys.readouterr().out

    def test_stats_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        assert cache_stats() is False
        assert "cache stats" in capsys.readouterr().out

    def test_backup_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # backup reads the source through the online-backup API, so a corrupt source
        # surfaces as a clean failure line, not a traceback.
        assert cache_backup() is False
        assert "cache backup failed" in capsys.readouterr().out


class TestActiveRunGuard:
    """Finding #5: destructive commands refuse while a run holds the lock."""

    def test_remove_refuses_while_a_run_holds_the_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        # Holding the single-instance lock on the data dir simulates an active run;
        # remove must refuse and leave the live db untouched.
        with single_instance_lock(str(tmp_path)):
            assert cache_remove() is False
        assert (tmp_path / "cache.db").exists()

    def test_restore_refuses_while_a_run_holds_the_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        with single_instance_lock(str(tmp_path)):
            assert cache_restore() is False
        # Refused before touching anything: the backup is still there to restore.
        assert (tmp_path / "cache.backup.db").exists()
        assert (tmp_path / "cache.db").exists()


class TestMissingFilesAreReportedNotRaised:
    """A missing cache/backup file echoes one line and returns False, no traceback."""

    def test_each_cache_command_reports_a_missing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))

        for command in (cache_backup, cache_remove, cache_stats, cache_check, cache_restore):
            assert command() is False
            assert capsys.readouterr().out.count("No file at") == 1

    def test_failed_backup_leaves_no_partial_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # A torn snapshot must not survive a failed backup: a later restore would
        # move it over the live database.
        assert cache_backup() is False
        assert "cache backup failed" in capsys.readouterr().out
        assert not (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.backup.db.tmp").exists()

    def test_failed_backup_preserves_the_previous_good_backup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        # The live db going corrupt is the very scenario backups exist for; the
        # failed re-backup must leave the good snapshot restorable.
        (tmp_path / "cache.db").write_text("not a database")
        assert cache_backup() is False

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
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        config = tmp_path / "config.yml"

        assert config_init() is True
        assert config.exists()

        # A filled-in config must survive an accidental re-run.
        config.write_text("sonarr: {url: http://mine}")
        assert config_init() is False
        assert config.read_text() == "sonarr: {url: http://mine}"
        assert "--force" in capsys.readouterr().out

        assert config_init(force=True) is True
        assert config.read_text() == Path(template_path()).read_text()


class TestRunSingleSelection:
    def test_no_selection_prints_a_hint_and_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The guard must fire before paths/logger setup: a usage error that
        # created a data dir or rotated log files would be a regression.
        def fail_resolve() -> NoReturn:
            raise AssertionError("resolve_paths must not be reached on a usage error")

        monkeypatch.setattr("seadexarr.modules.cli.resolve_paths", fail_resolve)

        assert run_single() is False
        assert "--radarr" in capsys.readouterr().out


class TestScheduleHours:
    """SCHEDULE_TIME parses leniently: bad values fall back to 6 with a report."""

    def test_unset_uses_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        assert _schedule_hours() == 6.0

    def test_a_valid_value_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", "0.5")
        assert _schedule_hours() == 0.5

    @pytest.mark.parametrize("raw", ["banana", "0", "-3", "inf", "nan"])
    def test_bad_values_fall_back_to_the_default(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", raw)
        assert _schedule_hours() == 6.0
        assert "Invalid SCHEDULE_TIME" in capsys.readouterr().out


class TestExitCodes:
    """The result callback maps a False return to exit code 1 (typer ignores returns)."""

    def test_failure_exits_nonzero_and_success_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        runner = CliRunner()

        # No cache.db yet: the command reports the missing file and must exit 1.
        result = runner.invoke(seadexarr_cli, ["cache", "stats"])
        assert result.exit_code == 1
        assert "No file at" in result.output

        _build_cache(tmp_path)
        result = runner.invoke(seadexarr_cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "entries=" in result.output
