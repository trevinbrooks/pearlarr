# pyright: strict
"""Tests for the unified data-directory resolver and its CLI surface.

Pins the behaviours the rest of the app relies on:

* ``resolve_paths`` honours the precedence ``--data-dir`` arg > ``SEADEXARR_DATA_DIR``
  env > the OS-standard ``platformdirs`` default, and lays every file under one dir.
* The global ``--data-dir`` flag folds into the env so each command (called directly in
  tests, not via ``ctx.obj``) sees it, and the flag wins over a pre-set env.
* Logs route to the resolved ``log_dir`` rather than the current working directory.
"""

import logging
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from seadexarr.modules.cli import seadexarr_cli
from seadexarr.modules.log import setup_logger
from seadexarr.modules.paths import APP_NAME, ensure_data_dir, resolve_paths

# These tests own the SEADEXARR_DATA_DIR env directly (precedence / default cases),
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
        monkeypatch.setenv("SEADEXARR_DATA_DIR", str(tmp_path / "from_env"))
        assert resolve_paths(str(tmp_path / "from_arg")).data_dir == str(tmp_path / "from_arg")

    def test_env_wins_over_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEADEXARR_DATA_DIR", str(tmp_path))
        assert resolve_paths().data_dir == str(tmp_path)

    def test_default_falls_back_to_platformdirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEADEXARR_DATA_DIR", raising=False)
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
        monkeypatch.setenv("SEADEXARR_DATA_DIR", str(tmp_path))
        result = runner.invoke(seadexarr_cli, ["paths"])
        assert result.exit_code == 0
        assert f"data_dir:    {tmp_path}" in result.output

    def test_data_dir_flag_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # monkeypatch records the key so the callback's os.environ write is restored
        # on teardown and never leaks into other tests.
        monkeypatch.setenv("SEADEXARR_DATA_DIR", str(tmp_path / "from_env"))
        result = runner.invoke(seadexarr_cli, ["--data-dir", str(tmp_path / "from_flag"), "paths"])
        assert result.exit_code == 0
        assert f"data_dir:    {tmp_path / 'from_flag'}" in result.output


class TestLogRouting:
    def test_logs_route_to_log_dir_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Run from an empty cwd so a stray "logs/" there would be unambiguous.
        monkeypatch.chdir(tmp_path)
        log_dir = str(tmp_path / "data" / "logs")

        logger = setup_logger(log_level="INFO", log_dir=log_dir)
        logger.info("routed")
        logging.shutdown()

        assert os.path.isfile(os.path.join(log_dir, "SeaDexArr.log"))
        assert not os.path.exists(tmp_path / "logs")
        # This test is about file routing; setup_logger also attaches a real console
        # handler, so drain it to keep the line off the terminal under `-s`.
        capsys.readouterr()

    def test_error_level_is_honored(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # ERROR is a first-class level now (it used to warn-and-default to INFO).
        log_dir = str(tmp_path / "logs")

        logger = setup_logger(log_level="ERROR", log_dir=log_dir)
        logger.error("kept")

        assert logger.level == logging.ERROR
        for handler in logger.handlers:
            handler.flush()
        content = Path(log_dir, "SeaDexArr.log").read_text(encoding="utf-8")
        assert "Invalid log level" not in content
        assert "kept" in content
        # This test is about the log file; drain setup_logger's console handler so its
        # line stays off the terminal under `-s`.
        capsys.readouterr()

    def test_invalid_log_level_complaint_reaches_the_file_log(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The complaint used to fire before the handlers were attached, so it only
        # reached logging.lastResort (stderr) - never the file log it warns about.
        log_dir = str(tmp_path / "logs")

        logger = setup_logger(log_level="BOGUS", log_dir=log_dir)

        assert logger.level == logging.INFO  # fell back exactly as before
        for handler in logger.handlers:
            handler.flush()
        content = Path(log_dir, "SeaDexArr.log").read_text(encoding="utf-8")
        assert "Invalid log level 'BOGUS'" in content
        # This test is about the log file; drain setup_logger's console handler so the
        # complaint stays off the terminal under `-s`.
        capsys.readouterr()
