"""Tests for the ``seadexarr cache`` CLI commands.

Pins the two behaviours the commands must guarantee around the SQLite cache:

* A corrupt / not-a-database ``cache.db`` is *reported*, never crashed on - the
  ``stats`` and ``check`` diagnostics (and ``backup``) return a clean False instead
  of letting a ``sqlite3`` traceback escape (finding #4). ``check`` exists to report
  bad integrity, so it must survive the very corruption it diagnoses.
* The destructive commands (``restore`` / ``remove``) take the single-instance run
  lock first and refuse while a run is active, so they never unlink or replace the
  live db out from under it (finding #5).

Each test points ``_paths()`` at its own ``tmp_path`` via ``SEADEX_ARR_DATA_DIR``
and calls the command functions directly (they return ``bool``).
"""

from seadexarr.modules.cache import CacheStore
from seadexarr.modules.cli import (
    cache_backup,
    cache_check,
    cache_remove,
    cache_restore,
    cache_stats,
)
from seadexarr.modules.config import Arr
from seadexarr.modules.runlock import single_instance_lock


def _build_cache(tmp_path) -> None:
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
    def test_backup_then_restore_preserves_data(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_backup() is True
        assert (tmp_path / "cache.backup.db").exists()

        # A stale WAL left next to cache.db must not shadow the restored snapshot;
        # restore clears the sidecars before moving the backup into place.
        (tmp_path / "cache.db-wal").write_text("stale")

        assert cache_restore() is True
        assert not (tmp_path / "cache.backup.db").exists()  # consumed by the move
        assert not (tmp_path / "cache.db-wal").exists()  # stale sidecar cleared

        # The restored db still holds the original entry.
        store = CacheStore.open_readonly(str(tmp_path / "cache.db"))
        try:
            assert store.get_cached_name(Arr.SONARR, 7) == "X"
        finally:
            store.close()

    def test_remove_deletes_db_and_sidecars(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        (tmp_path / "cache.db-wal").write_text("stale")
        (tmp_path / "cache.db-shm").write_text("stale")

        assert cache_remove() is True
        assert not (tmp_path / "cache.db").exists()
        assert not (tmp_path / "cache.db-wal").exists()
        assert not (tmp_path / "cache.db-shm").exists()


class TestHealthyDiagnostics:
    def test_stats_and_check_report_a_healthy_db(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_stats() is True
        assert cache_check() is True
        assert "integrity: ok" in capsys.readouterr().out


class TestCorruptDatabaseIsReportedNotCrashed:
    """Finding #4: a corrupt cache.db is reported cleanly, never tracebacks."""

    def test_check_on_corrupt_db_returns_false_without_raising(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # Reporting bad integrity is this command's whole job, so it must not crash
        # on the very corruption it diagnoses.
        assert cache_check() is False
        assert "integrity" in capsys.readouterr().out

    def test_stats_on_corrupt_db_returns_false_without_raising(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        assert cache_stats() is False
        assert "cache stats" in capsys.readouterr().out

    def test_backup_on_corrupt_db_returns_false_without_raising(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # backup reads the source through the online-backup API, so a corrupt source
        # surfaces as a clean failure line, not a traceback.
        assert cache_backup() is False
        assert "cache backup failed" in capsys.readouterr().out


class TestActiveRunGuard:
    """Finding #5: destructive commands refuse while a run holds the lock."""

    def test_remove_refuses_while_a_run_holds_the_lock(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        # Holding the single-instance lock on the data dir simulates an active run;
        # remove must refuse and leave the live db untouched.
        with single_instance_lock(str(tmp_path)):
            assert cache_remove() is False
        assert (tmp_path / "cache.db").exists()

    def test_restore_refuses_while_a_run_holds_the_lock(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        with single_instance_lock(str(tmp_path)):
            assert cache_restore() is False
        # Refused before touching anything: the backup is still there to restore.
        assert (tmp_path / "cache.backup.db").exists()
        assert (tmp_path / "cache.db").exists()
