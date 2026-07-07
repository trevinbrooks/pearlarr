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
* Failure paths report cleanly (missing files echo one hint line, not a traceback),
  failure text goes to stderr (so `seadexarr config show > cfg.yml` stays clean)
  while success output stays on stdout, and a False return maps to exit code 1
  through the Typer apps' result callback.
* ``config init`` never overwrites a filled-in config.yml without ``--force``.
* ``run single`` with no selection flag runs every *configured* arr (scheduled-mode
  symmetry); an explicit flag for an unconfigured arr refuses cleanly, an implicit
  selection skips it with a dim ledger note - except a half-configured arr (url
  without api_key), whose skip warns by name (``_configured_arrs``).
* The inspection commands (``config validate`` / ``config show``) never write the
  starter template, and ``show`` masks secret-named values (plus the free-form
  ``qbittorrent.options`` block and URL-embedded logins) while keeping unset
  ones ``null``.
* ``advanced.log_level`` is applied to CLI runs as soon as the config is read,
  and ``--log-level`` overrides it (cli > config).

Each test points ``resolve_paths()`` at its own ``tmp_path`` via ``SEADEX_ARR_DATA_DIR``
and calls the command functions directly (they return ``bool``); the exit-code
tests go through ``CliRunner`` since the callback only runs inside typer.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from seadexarr.modules.cache import CacheStore
from seadexarr.modules.cli import (
    _configured_arrs,
    _schedule_hours,
    cache_backup,
    cache_check,
    cache_remove,
    cache_restore,
    cache_stats,
    config_init,
    config_show,
    config_validate,
    run_single,
    seadexarr_cli,
)
from seadexarr.modules.config import AppConfig, Arr, template_path
from seadexarr.modules.log import (
    LogLevel,
    RichConsoleHandler,
    StyledLine,
    apply_log_level,
    console_payload,
    setup_logger,
)
from seadexarr.modules.manual_import import ImportWaitMode
from seadexarr.modules.paths import AppPaths, resolve_paths
from seadexarr.modules.runlock import single_instance_lock

from .fakes import CaptureHandler


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
    def test_backup_then_restore_preserves_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        assert cache_backup() is True
        assert (tmp_path / "cache.backup.db").exists()
        # Success is confirmed out loud (on stdout), not silently.
        assert f"Backed up cache to {tmp_path / 'cache.backup.db'}." in capsys.readouterr().out

        # A stale WAL left next to cache.db must not shadow the restored snapshot;
        # restore clears the sidecars before swapping the copy into place.
        (tmp_path / "cache.db-wal").write_text("stale")

        assert cache_restore() is True
        assert f"Restored cache from {tmp_path / 'cache.backup.db'}." in capsys.readouterr().out
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
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        (tmp_path / "cache.db-wal").write_text("stale")
        (tmp_path / "cache.db-shm").write_text("stale")

        assert cache_remove() is True
        assert f"Removed {tmp_path / 'cache.db'}." in capsys.readouterr().out
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
        # on the very corruption it diagnoses. The failure line goes to stderr.
        assert cache_check() is False
        assert "integrity" in capsys.readouterr().err

    def test_stats_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        assert cache_stats() is False
        assert "cache stats" in capsys.readouterr().err

    def test_backup_on_corrupt_db_returns_false_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # backup reads the source through the online-backup API, so a corrupt source
        # surfaces as a clean failure line (on stderr), not a traceback.
        assert cache_backup() is False
        assert "cache backup failed" in capsys.readouterr().err


class TestActiveRunGuard:
    """Finding #5: destructive commands refuse while a run holds the lock."""

    def test_remove_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)

        # Holding the single-instance lock on the data dir simulates an active run;
        # remove must refuse and leave the live db untouched.
        with single_instance_lock(str(tmp_path)):
            assert cache_remove() is False
        assert (tmp_path / "cache.db").exists()
        # The refusal is a user-facing stderr line, not a silent no-op (capturing
        # it also keeps it off the terminal under `-s`).
        assert "another seadexarr run is active" in capsys.readouterr().err.lower()

    def test_restore_refuses_while_a_run_holds_the_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        with single_instance_lock(str(tmp_path)):
            assert cache_restore() is False
        # Refused before touching anything: the backup is still there to restore.
        assert (tmp_path / "cache.backup.db").exists()
        assert (tmp_path / "cache.db").exists()
        # The refusal is a user-facing stderr line, not a silent no-op (capturing
        # it also keeps it off the terminal under `-s`).
        assert "another seadexarr run is active" in capsys.readouterr().err.lower()


class TestMissingFilesAreReportedNotRaised:
    """A missing cache/backup file echoes one hinting stderr line and returns False."""

    def test_each_cache_command_reports_a_missing_file_with_a_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))

        # Each message names what's missing and hints at how to get one, so a
        # fresh install isn't met with a bare "No file at ...".
        checks: list[tuple[Callable[[], bool], str]] = [
            (cache_backup, "No cache database at"),
            (cache_stats, "No cache database at"),
            (cache_check, "No cache database at"),
            (cache_backup, "it is created by the first run"),
            (cache_remove, "nothing to remove"),
            (cache_restore, "No backup at"),
            (cache_restore, "run 'seadexarr cache backup' first"),
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
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        (tmp_path / "cache.db").write_text("not a database")

        # A torn snapshot must not survive a failed backup: a later restore would
        # move it over the live database.
        assert cache_backup() is False
        assert "cache backup failed" in capsys.readouterr().err
        assert not (tmp_path / "cache.backup.db").exists()
        assert not (tmp_path / "cache.backup.db.tmp").exists()

    def test_failed_backup_preserves_the_previous_good_backup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        _build_cache(tmp_path)
        assert cache_backup() is True

        # The live db going corrupt is the very scenario backups exist for; the
        # failed re-backup must leave the good snapshot restorable.
        (tmp_path / "cache.db").write_text("not a database")
        assert cache_backup() is False
        # The failure line itself is pinned by test_failed_backup_leaves_no_partial_snapshot;
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
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        config = tmp_path / "config.yml"

        assert config_init() is True
        assert config.exists()

        # A filled-in config must survive an accidental re-run; the refusal is
        # a stderr line.
        config.write_text("sonarr: {url: http://mine}")
        assert config_init() is False
        assert config.read_text() == "sonarr: {url: http://mine}"
        assert "--force" in capsys.readouterr().err

        assert config_init(force=True) is True
        assert config.read_text() == Path(template_path()).read_text(encoding="utf-8")
        # The write is proven by the file content above; drain the post-force echo so
        # it can't spill to the terminal under `-s` (output after a mid-test
        # readouterr() is flushed there at teardown when global capture is off).
        capsys.readouterr()


class _RunArrsRecorder:
    """A stand-in for ``cli._run_arrs`` that records how ``run_single`` calls it."""

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
    """No selection flag = every arr, implicitly; any flag/id narrows explicitly.

    ``_run_arrs`` is faked out, so these pin only the selection wiring: which
    ``(arr, item_id)`` pairs are requested and whether the request counts as
    explicit (an explicit request for an unconfigured arr must refuse rather
    than skip - that arm is pinned on ``_configured_arrs`` below).
    """

    @pytest.fixture
    def recorder(self, monkeypatch: pytest.MonkeyPatch) -> _RunArrsRecorder:
        recorder = _RunArrsRecorder()
        monkeypatch.setattr("seadexarr.modules.cli._run_arrs", recorder)
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
    def capture(self, logger: logging.Logger) -> CaptureHandler:
        handler = CaptureHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return handler

    def test_implicit_selection_skips_the_unconfigured_arr(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        kept = _configured_arrs(
            [(Arr.RADARR, None), (Arr.SONARR, None)],
            _config_with(sonarr=True),
            explicit=False,
            config_path="config.yml",
            logger=logger,
        )
        assert kept == [(Arr.SONARR, None)]
        skip = next(r for r in capture.records if "adarr" in r.getMessage())
        assert skip.levelno == logging.INFO  # a Sonarr-only setup is normal, not an error
        # The note is styled to sit inside the boot ledger it lands in: indented,
        # dimmed like the ledger's own secondary lines, not a bare column-0 line.
        # The arr name is capitalized prose, not a lowercase config key.
        assert skip.getMessage() == "  Radarr not configured - skipped"
        assert console_payload(skip) == StyledLine(style="grey50")

    def test_a_half_configured_arr_warns_by_name(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        config = AppConfig.model_validate(
            {"sonarr": {"url": "http://sonarr:8989", "api_key": "k"}, "radarr": {"url": "http://radarr:7878"}},
        )
        kept = _configured_arrs(
            [(Arr.RADARR, None), (Arr.SONARR, None)],
            config,
            explicit=False,
            config_path="config.yml",
            logger=logger,
        )
        assert kept == [(Arr.SONARR, None)]
        # url XOR api_key is almost certainly an accident: the skip must be loud
        # and name the missing half, not read like an intentional single-arr setup.
        # Dotted keys stay lowercase; the prose subject is capitalized.
        warning = next(r for r in capture.records if r.levelno == logging.WARNING)
        assert warning.getMessage() == "radarr.url is set but radarr.api_key is not - skipping Radarr"

    def test_explicit_selection_of_a_half_configured_arr_names_only_the_missing_key(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        config = AppConfig.model_validate({"radarr": {"url": "http://radarr:7878"}})
        kept = _configured_arrs([(Arr.RADARR, None)], config, explicit=True, config_path="config.yml", logger=logger)
        assert kept is None
        error = next(r for r in capture.records if r.levelno == logging.ERROR)
        assert "set radarr.api_key in config.yml" in error.getMessage()

    def test_explicit_selection_of_an_unconfigured_arr_refuses(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        kept = _configured_arrs(
            [(Arr.RADARR, 42)],
            _config_with(sonarr=True),
            explicit=True,
            config_path="config.yml",
            logger=logger,
        )
        assert kept is None
        error = next(r for r in capture.records if r.levelno == logging.ERROR)
        assert "radarr.url" in error.getMessage()
        # Prose subject capitalized; the dotted keys above stay lowercase.
        assert error.getMessage().startswith("Radarr was selected but is not configured")

    def test_nothing_configured_reports_and_refuses(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        kept = _configured_arrs(
            [(Arr.RADARR, None), (Arr.SONARR, None)],
            _config_with(),
            explicit=False,
            config_path="config.yml",
            logger=logger,
        )
        assert kept is None
        assert any(r.levelno == logging.ERROR for r in capture.records)

    def test_fully_configured_passes_through_unchanged(
        self,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        arrs: list[tuple[Arr, int | None]] = [(Arr.RADARR, None), (Arr.SONARR, 7)]
        assert (
            _configured_arrs(
                arrs, _config_with(sonarr=True, radarr=True), explicit=True, config_path="config.yml", logger=logger
            )
            == arrs
        )
        assert capture.records == []


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
                # The failing URL (from arrapi's message) is surfaced, so the
                # user sees which connection broke, not just which leg was running.
                "Sonarr run failed: Failed to Connect to http://sonarr:8989 - check sonarr.url",
                id="arr-unreachable",
            ),
            pytest.param("unauthorized", "Sonarr rejected the API key - check sonarr.api_key", id="arr-unauthorized"),
        ],
    )
    def test_a_failed_arr_leg_reports_cleanly_and_returns_false(
        self,
        error: str,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from arrapi.exceptions import ConnectionFailure, Unauthorized

        from seadexarr.modules.run_services import QbitConnectionError, RunDeps

        exceptions: dict[str, Exception] = {
            "qbit": QbitConnectionError("qBittorrent connection failed - check the host and credentials"),
            "unreachable": ConnectionFailure("Failed to Connect to http://sonarr:8989"),
            "unauthorized": Unauthorized("Invalid apikey"),
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
        # so an arrapi failure there must not be pinned on Sonarr alone: a Radarr
        # outage would otherwise read "check sonarr.url" - confidently wrong.
        from arrapi.exceptions import ConnectionFailure, Unauthorized

        from seadexarr.modules.run_services import RunDeps

        exceptions: dict[str, Exception] = {
            "unreachable": ConnectionFailure("Failed to Connect to http://radarr:7878"),
            "unauthorized": Unauthorized("Invalid apikey"),
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
        assert "line 3" in out  # the position survives, so the user can find it
        assert "hunter2" not in out
        assert "Traceback" not in out

    def test_a_mapping_source_failure_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import seadexarr.modules.mappings as mappings_mod

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
        # (the refresh path falls open to the cached copy; a first download has
        # none): urlopen raises URLError (an OSError), which must surface as a
        # one-line hint, not a ten-frame traceback.
        from urllib.error import URLError

        import seadexarr.modules.mappings as mappings_mod

        def fail_resolver(*args: object, **kwargs: object) -> NoReturn:
            raise URLError("no route to host")

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
    def test_nothing_configured_fails_before_the_mapping_fetch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A run with nothing to do must fail fast on the selection check, not
        # download/parse the (large) mapping sources first.
        def fail_build(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError("mapping sources must not be fetched when nothing can run")

        monkeypatch.setattr("seadexarr.modules.cli._build_resolver", fail_build)
        paths = resolve_paths()
        os.makedirs(paths.data_dir)
        Path(paths.config).write_text("", encoding="utf-8")  # valid config, neither arr configured

        assert run_single() is False
        capsys.readouterr()  # swallow the boot banner + error line


class TestConfigInspection:
    """``config validate`` / ``config show``: report cleanly, never write files."""

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
            "  username: admin\n"
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
        assert "hunter2" not in out
        assert "admin" not in out
        assert "basicpass" not in out
        assert "url: http://REDACTED@sonarr:8989" in out

    def test_show_missing_file_fails_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The failure hint is stderr-only: `config show > cfg.yml` must not
        # capture error text as if it were config.
        assert config_show() is False
        captured = capsys.readouterr()
        assert "config init" in captured.err
        assert captured.out == ""


class TestVersionAndHelp:
    def test_version_flag_prints_the_package_version(self) -> None:
        result = CliRunner().invoke(seadexarr_cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("seadexarr ")

    def test_short_help_alias_works(self) -> None:
        result = CliRunner().invoke(seadexarr_cli, ["-h"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_bare_group_shows_its_help(self) -> None:
        # no_args_is_help: a bare `seadexarr run` teaches instead of erroring opaquely.
        result = CliRunner().invoke(seadexarr_cli, ["run"])
        assert "scheduled" in result.output
        assert "single" in result.output


class TestApplyLogLevel:
    def test_repoints_logger_and_console_thresholds(self, tmp_path: Path) -> None:
        logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"))
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
    """The headline fix: ``advanced.log_level`` reaches CLI runs, ``--log-level`` wins.

    ``apply_log_level`` itself is pinned above; these pin that ``_run_arrs``
    actually calls it once the config is readable, with cli > config precedence
    (the original bug: the config level was dead on every CLI run).
    """

    @pytest.fixture
    def applied(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def record(logger: logging.Logger, log_level: str) -> None:
            calls.append(log_level)

        monkeypatch.setattr("seadexarr.modules.cli.apply_log_level", record)
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


class TestScheduleHours:
    """Cadence precedence: valid SCHEDULE_TIME env (deprecated) > config > 6.

    The SCHEDULE_TIME notices go through the logger (its only caller runs after
    ``setup_logger``), so they reach the log file and render styled - pinned
    here via a capture handler rather than stdout.
    """

    @pytest.fixture
    def capture(self, logger: logging.Logger) -> CaptureHandler:
        handler = CaptureHandler()
        logger.addHandler(handler)
        return handler

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
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        missing = tmp_path / "config.yml"
        assert _schedule_hours(str(missing), logger) == 6.0
        # No template-copy side effect: _load_shared_config owns the first-run copy.
        assert not missing.exists()
        assert capture.records == []  # nothing to warn about without the env var

    def test_unset_env_reads_the_config_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        assert _schedule_hours(self._write_config(tmp_path, 2.5), logger) == 2.5

    def test_unreadable_config_falls_back_to_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        path = tmp_path / "config.yml"
        path.write_text("schedule:\n  interval_hours: -1\n")  # fails validation
        assert _schedule_hours(str(path), logger) == 6.0

    def test_a_valid_env_value_wins_over_config_with_deprecation_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", "0.5")
        assert _schedule_hours(self._write_config(tmp_path, 2.5), logger) == 0.5
        notice = next(r for r in capture.records if "deprecated" in r.getMessage())
        assert notice.levelno == logging.WARNING

    @pytest.mark.parametrize("raw", ["banana", "0", "-3", "inf", "nan"])
    def test_bad_env_values_fall_through_to_config(
        self,
        raw: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", raw)
        assert _schedule_hours(self._write_config(tmp_path, 2.5), logger) == 2.5
        notice = next(r for r in capture.records if r.levelno == logging.WARNING)
        assert "Invalid SCHEDULE_TIME" in notice.getMessage()
        # The notice reports the value actually used, not a claim about its source.
        assert "using 2.5 hours" in notice.getMessage()

    def test_bad_env_with_missing_config_reports_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
        capture: CaptureHandler,
    ) -> None:
        monkeypatch.setenv("SCHEDULE_TIME", "banana")
        assert _schedule_hours(str(tmp_path / "config.yml"), logger) == 6.0
        assert any("using 6 hours" in r.getMessage() for r in capture.records)

    def test_malformed_yaml_falls_back_to_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        logger: logging.Logger,
    ) -> None:
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        path = tmp_path / "config.yml"
        path.write_text("schedule: [unclosed\n")
        assert _schedule_hours(str(path), logger) == 6.0


class TestExitCodes:
    """The result callback maps a False return to exit code 1 (typer ignores returns)."""

    def test_failure_exits_nonzero_and_success_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
        runner = CliRunner()

        # No cache.db yet: the command reports the missing file (on stderr,
        # keeping stdout clean for scripts) and must exit 1.
        result = runner.invoke(seadexarr_cli, ["cache", "stats"])
        assert result.exit_code == 1
        assert "No cache database at" in result.stderr
        assert result.stdout == ""

        _build_cache(tmp_path)
        result = runner.invoke(seadexarr_cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "entries:" in result.stdout
