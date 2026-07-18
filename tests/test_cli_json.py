# pyright: strict
"""The subcommands' `--json` seat: envelope shape, stream separation, no hub leak.

Every subcommand wraps its body in `cli_surface`, emitting the same typed events
run commands do. With `--json` the JSON seat writes one envelope line per event
to stdout (errors included, at level ERROR), stderr stays empty, and the exit
semantics (the returned bool) are unchanged. The human-output byte-parity is
pinned in test_cli.py. Here we pin the machine surface and that the surface is
torn down cleanly after each command (`current_hub()` back to the default).
"""

import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from pearlarr.cache import CacheStore
from pearlarr.cli import (
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
    show_paths,
)
from pearlarr.config import Arr
from pearlarr.config_migrations import CONFIG_VERSION
from pearlarr.output import Diagnostic, Severity, emit_to_hub
from pearlarr.output.runtime import current_hub

_ENVELOPE = ["schema_version", "time", "event", "level", "message"]


def _build_cache(tmp_path: Path) -> None:
    """Write a real on-disk cache.db under tmp_path holding one entry."""

    store = CacheStore.load(str(tmp_path / "cache.db"), config_checksum="x")
    store.update_cache(Arr.SONARR, 7, {"name": "X"})
    store.save(preview=False)
    store.close()


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yml").write_text(body, encoding="utf-8")


def _envelope_lines(out: str) -> list[dict[str, object]]:
    """Parse every stdout line as a JSON envelope, pinning the first five keys."""

    parsed: list[dict[str, object]] = []
    for line in out.splitlines():
        obj = cast("dict[str, object]", json.loads(line))
        assert list(obj)[:5] == _ENVELOPE
        parsed.append(obj)
    return parsed


def _obj(value: object) -> dict[str, object]:
    """Assert a parsed JSON value is an object and give it a concrete element type."""

    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path))


class TestJsonSuccessArms:
    """Each command's success arm emits its event as one envelope line, stderr empty."""

    def test_paths(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert show_paths(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "paths_shown"
        assert obj["config"] == str(tmp_path / "config.yml")
        assert captured.err == ""

    def test_config_init(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert config_init(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "starter_config_written"
        assert captured.err == ""

    def test_config_validate(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_config(tmp_path, "sonarr:\n  url: http://s\n  api_key: k\n")
        assert config_validate(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "config_validated"
        assert obj["sonarr_missing_keys"] == []
        assert obj["radarr_missing_keys"] == ["radarr.url", "radarr.api_key"]
        assert captured.err == ""

    def test_config_migrate_old_schema(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_config(tmp_path, "seadex:\n  public_only: true\nsonarr:\n  url: http://s\n")
        assert config_migrate(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "config_migrated"
        assert obj["backup_path"] == str(tmp_path / "config.yml.bak")
        assert captured.err == ""

    def test_config_migrate_current_is_nothing_to_do(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_config(tmp_path, f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://s\n")
        assert config_migrate(json_output=True) is True
        (obj,) = _envelope_lines(capsys.readouterr().out)
        assert obj["event"] == "config_up_to_date"

    def test_config_show(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_config(tmp_path, "sonarr:\n  url: http://s\n  api_key: k\n")
        assert config_show(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "effective_config_shown"
        assert _obj(obj["config"])["sonarr"] is not None
        assert captured.err == ""

    def test_cache_backup(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _build_cache(tmp_path)
        assert cache_backup(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "cache_backed_up"
        assert captured.err == ""

    def test_cache_restore(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _build_cache(tmp_path)
        assert cache_backup(json_output=True) is True
        capsys.readouterr()
        assert cache_restore(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "cache_restored"
        assert captured.err == ""

    def test_cache_remove(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _build_cache(tmp_path)
        assert cache_remove(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "cache_removed"
        assert captured.err == ""

    def test_cache_stats(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _build_cache(tmp_path)
        assert cache_stats(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "cache_stats_reported"
        assert obj["entries"] == 1
        assert captured.err == ""

    def test_cache_check(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _build_cache(tmp_path)
        assert cache_check(json_output=True) is True
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "cache_integrity_reported"
        assert obj["result"] == "ok"
        assert captured.err == ""


class TestJsonErrorArms:
    """One error arm per command family: the failure is an ERROR envelope on stdout, stderr empty."""

    def test_missing_cache_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        # No cache.db: the missing-file report is a diagnostic on the json stream.
        assert cache_stats(json_output=True) is False
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "diagnostic"
        assert obj["level"] == "ERROR"
        assert captured.err == ""

    def test_missing_config_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert config_validate(json_output=True) is False
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "diagnostic"
        assert obj["level"] == "ERROR"
        assert captured.err == ""

    def test_invalid_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_config(tmp_path, "sonar:\n  url: x\n")
        assert config_validate(json_output=True) is False
        captured = capsys.readouterr()
        (obj,) = _envelope_lines(captured.out)
        assert obj["event"] == "diagnostic"
        assert obj["level"] == "ERROR"
        message = obj["message"]
        assert isinstance(message, str)
        assert "Invalid configuration" in message
        assert captured.err == ""


class TestRedactionCanary:
    """`config show --json` masks secrets in the `config` object and anywhere in the stream."""

    def test_secrets_are_redacted_in_the_json_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_config(
            tmp_path,
            "sonarr:\n  url: http://sonarr:8989\n  api_key: hunter2\n"
            "notifications:\n  discord_url: https://discord.com/api/webhooks/1/tok\n",
        )
        assert config_show(json_output=True) is True
        out = capsys.readouterr().out

        (obj,) = _envelope_lines(out)
        sonarr = _obj(_obj(obj["config"])["sonarr"])
        assert sonarr["api_key"] == "REDACTED"
        # The raw secrets never appear anywhere in the emitted stream.
        assert "hunter2" not in out
        assert "tok" not in out


class TestHumanModeLeavesNoHub:
    """A command restores the process default hub on exit. A later emit can't leak into its seat."""

    def test_command_uninstalls_its_hub(self, capsys: pytest.CaptureFixture[str]) -> None:
        before = current_hub()
        assert show_paths() is True  # human mode
        capsys.readouterr()  # drain the paths block

        # The default (renderer-less) hub is back, so a later emit reaches no seat.
        assert current_hub() is before
        emit_to_hub(Diagnostic(severity=Severity.ERROR, message="leak check", origin="test"))
        leaked = capsys.readouterr()
        assert leaked.out == ""
        assert leaked.err == ""


class TestCliRunnerJsonLeg:
    """End-to-end through typer: exit code + stream separation with `--json`."""

    def test_cache_stats_json_exit_and_streams(self, tmp_path: Path) -> None:
        _build_cache(tmp_path)
        result = CliRunner().invoke(pearlarr_cli, ["cache", "stats", "--json"])
        assert result.exit_code == 0
        (obj,) = _envelope_lines(result.stdout)
        assert obj["event"] == "cache_stats_reported"
        assert result.stderr == ""

    def test_missing_cache_json_exits_one_on_stdout(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["cache", "stats", "--json"])
        assert result.exit_code == 1
        (obj,) = _envelope_lines(result.stdout)
        assert obj["level"] == "ERROR"
        assert result.stderr == ""
