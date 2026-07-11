# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the log-routing tests wire the real (private) cli hub installer.
"""Tests for the unified data-directory resolver and its CLI surface.

Pins the behaviors the rest of the app relies on:

* `resolve_paths` honors the precedence `--data-dir` arg > `PEARLARR_DATA_DIR`
  env > the OS-standard `platformdirs` default, and lays every file under one dir.
* The global `--data-dir` flag folds into the env so each command (called directly in
  tests, not via `ctx.obj`) sees it, and the flag wins over a pre-set env.
* Logs route to the resolved `log_dir` rather than the current working directory.
"""

import io
import logging
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pearlarr.modules.cli import _install_output_hub, pearlarr_cli
from pearlarr.modules.log import setup_logger
from pearlarr.modules.paths import APP_NAME, ensure_data_dir, resolve_paths

# These tests own the PEARLARR_DATA_DIR env directly (precedence / default cases),
# so they opt out of the autouse tmp data-dir isolation (see tests/conftest.py).
pytestmark = pytest.mark.real_data_dir

runner = CliRunner()


class TestResolvePaths:
    def test_every_member_lives_under_the_data_dir(self, tmp_path: Path) -> None:
        paths = resolve_paths(str(tmp_path))
        base = str(tmp_path)
        assert paths.data_dir == base
        assert paths.config == os.path.join(base, "config.yml")
        assert paths.cache == os.path.join(base, "cache.db")
        assert paths.cache_backup == os.path.join(base, "cache.backup.db")
        assert paths.mappings_db == os.path.join(base, "mappings.db")
        assert paths.log_dir == os.path.join(base, "logs")

    def test_arg_wins_over_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path / "from_env"))
        assert resolve_paths(str(tmp_path / "from_arg")).data_dir == str(tmp_path / "from_arg")

    def test_env_wins_over_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        assert resolve_paths().data_dir == str(tmp_path)

    def test_default_falls_back_to_platformdirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PEARLARR_DATA_DIR", raising=False)
        # The OS-standard per-user location; we don't pin the prefix (it differs per
        # platform), only that it is absolute and names the app.
        data_dir = resolve_paths().data_dir
        assert os.path.isabs(data_dir)
        assert os.path.basename(data_dir) == APP_NAME

    def test_relative_path_is_absolutised(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_paths("reldir").data_dir == str(tmp_path / "reldir")


class TestEnsureDataDir:
    def test_creates_the_missing_directory(self, tmp_path: Path) -> None:
        paths = resolve_paths(str(tmp_path / "nested" / "dir"))
        assert not os.path.exists(paths.data_dir)
        ensure_data_dir(paths)
        assert os.path.isdir(paths.data_dir)


class TestPathsCommand:
    def test_prints_the_resolved_data_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))
        result = runner.invoke(pearlarr_cli, ["paths"])
        assert result.exit_code == 0
        assert f"data_dir:    {tmp_path}" in result.output

    def test_data_dir_flag_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # monkeypatch records the key so the callback's os.environ write is restored
        # on teardown and never leaks into other tests.
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path / "from_env"))
        result = runner.invoke(pearlarr_cli, ["--data-dir", str(tmp_path / "from_flag"), "paths"])
        assert result.exit_code == 0
        assert f"data_dir:    {tmp_path / 'from_flag'}" in result.output


class TestLogRouting:
    """Log-file routing through the REAL cli wiring (hub + bridge, then setup_logger)."""

    @staticmethod
    def _install(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """The production install against `data_dir`; stdout swapped off the tty."""

        monkeypatch.setattr(sys, "stdout", io.StringIO())
        _install_output_hub(resolve_paths(str(data_dir)))
        return data_dir / "logs" / "Pearlarr.log"

    def test_logs_route_to_log_dir_not_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Run from an empty cwd so a stray "logs/" there would be unambiguous.
        monkeypatch.chdir(tmp_path)
        log_file = self._install(tmp_path / "data", monkeypatch)

        logger = setup_logger(log_level="INFO", console_format="plain")
        logger.info("routed")

        assert log_file.is_file()
        assert "routed" in log_file.read_text(encoding="utf-8")
        assert not os.path.exists(tmp_path / "logs")

    def test_error_level_is_honored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # ERROR is a first-class level now (it used to warn-and-default to INFO).
        log_file = self._install(tmp_path / "data", monkeypatch)

        logger = setup_logger(log_level="ERROR", console_format="plain")
        logger.error("kept")

        assert logger.level == logging.ERROR
        content = log_file.read_text(encoding="utf-8")
        assert "Invalid log level" not in content
        assert "kept" in content

    def test_invalid_log_level_complaint_reaches_the_file_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The complaint fires inside setup_logger; with the hub + bridge already
        # installed (the cli order) it reaches the FileLogSink, never lastResort.
        log_file = self._install(tmp_path / "data", monkeypatch)

        logger = setup_logger(log_level="BOGUS", console_format="plain")

        assert logger.level == logging.INFO  # fell back exactly as before
        assert "Invalid log level 'BOGUS'" in log_file.read_text(encoding="utf-8")
