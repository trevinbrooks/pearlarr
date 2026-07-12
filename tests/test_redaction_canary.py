# pyright: strict
"""Canary-secret redaction: the SECURITY.md guarantee, driven surface by surface.

A config whose every secret field holds a unique canary string is pushed
through `config show`, `config validate`, a full (network-blocked) run's
console output (text and JSON) and file log, validation-error text, and the
notifier's failure warnings. No canary may appear in any of them, at any
log level.
"""

import os
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from pearlarr.arr_http import ArrConnectionError, ArrHttp
from pearlarr.cli import pearlarr_cli
from pearlarr.config import Arr
from pearlarr.manual_import import Outcome
from pearlarr.notify import Notifier
from pearlarr.paths import resolve_paths
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .fakes import diagnostic_messages, install_recording_hub

# One unique canary per secret-bearing config field, so a leak names its source.
CANARIES: dict[str, str] = {
    "sonarr api key": "CANARY-sonarr-key-3f9a1c",
    "radarr api key": "CANARY-radarr-key-77b2e0",
    "url-embedded login": "CANARY-url-pass-e0d97a",
    "qbittorrent username": "CANARY-qbit-user-91d24b",
    "qbittorrent password": "CANARY-qbit-pass-c44e8d",
    "options value": "CANARY-options-proxy-a1f70e",
    "discord webhook token": "CANARY-discord-token-8ac31f",
    "wait webhook token": "CANARY-wait-hook-5b6fd2",
}
DISCORD_URL = f"https://discord.example.invalid/api/webhooks/1234/{CANARIES['discord webhook token']}"
WEBHOOK_URL = f"https://ntfy.example.invalid/{CANARIES['wait webhook token']}"

# Port 9 (discard) on loopback: refused instantly, so nothing leaves the host
# even where a client library bypasses the respx-patched httpx transport.
CANARY_CONFIG = f"""
sonarr:
  url: http://admin:{CANARIES["url-embedded login"]}@127.0.0.1:9
  api_key: {CANARIES["sonarr api key"]}
radarr:
  url: http://127.0.0.1:9
  api_key: {CANARIES["radarr api key"]}
qbittorrent:
  host: http://127.0.0.1:9
  username: {CANARIES["qbittorrent username"]}
  password: {CANARIES["qbittorrent password"]}
  options:
    PROXIES: {CANARIES["options value"]}
notifications:
  discord_url: {DISCORD_URL}
  wait_webhook_url: {WEBHOOK_URL}
advanced:
  log_level: DEBUG
"""


def _assert_clean(text: str) -> None:
    """Fail naming the leaked field if any canary appears in `text`."""

    for field, canary in CANARIES.items():
        assert canary not in text, f"{field} leaked into output"


def _write_canary_config(*, log_format: str = "auto") -> None:
    paths = resolve_paths()
    os.makedirs(paths.data_dir, exist_ok=True)
    Path(paths.config).write_text(CANARY_CONFIG + f"  log_format: {log_format}\n", encoding="utf-8")


def _data_dir_text() -> str:
    """Every log file the run left in the data directory, concatenated."""

    paths = resolve_paths()
    logs: list[Path] = sorted(Path(paths.log_dir).rglob("*.log")) if os.path.isdir(paths.log_dir) else []
    return "\n".join(log.read_text(encoding="utf-8") for log in logs)


class TestCliSurfaces:
    """`config show`, `config validate`, and a failed run's console + file log all redact every canary secret."""

    def test_config_show_is_canary_free(self) -> None:
        _write_canary_config()
        result = CliRunner().invoke(pearlarr_cli, ["config", "show"])
        assert result.exit_code == 0
        assert "REDACTED" in result.output
        _assert_clean(result.output)

    def test_config_validate_is_canary_free(self) -> None:
        _write_canary_config()
        result = CliRunner().invoke(pearlarr_cli, ["config", "validate"])
        assert result.exit_code == 0
        _assert_clean(result.output)

    def test_validation_error_hides_a_secret_pasted_under_a_wrong_key(self) -> None:
        # A credential pasted under a mistyped/wrong-typed key must not be
        # echoed back by the validation error (hide_input_in_errors).
        paths = resolve_paths()
        os.makedirs(paths.data_dir, exist_ok=True)
        pasted = "CANARY-pasted-secret-0d11f3"
        Path(paths.config).write_text(f"advanced:\n  sleep_time: {pasted}\n", encoding="utf-8")
        result = CliRunner().invoke(pearlarr_cli, ["config", "validate"])
        assert result.exit_code == 1
        assert "sleep_time" in result.output
        assert pasted not in result.output

    def test_failed_run_console_and_file_log_are_canary_free(self) -> None:
        # Every outbound httpx call fails, so the run exercises its error
        # paths (the ones most tempted to interpolate URLs and credentials)
        # across the console and the always-on file log, at DEBUG level.
        _write_canary_config()
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=httpx.ConnectError("blocked by test"))
            result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1
        _assert_clean(result.output)
        log_text = _data_dir_text()
        assert log_text  # the always-on file log was actually written and scanned
        _assert_clean(log_text)

    def test_failed_run_json_stream_is_canary_free(self) -> None:
        # The JSON event stream is a documented machine interface, so it gets
        # its own leg of the same failed-run sweep.
        _write_canary_config(log_format="json")
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=httpx.ConnectError("blocked by test"))
            result = CliRunner().invoke(pearlarr_cli, ["run", "single"])
        assert result.exit_code == 1
        assert '"event":' in result.output  # the JSON surface actually rendered
        _assert_clean(result.output)


class TestEnvOverrideSurfaces:
    """A canary secret supplied via a `PEARLARR_*__*` environment override never leaks into config output."""

    ENV_KEY = "CANARY-env-key-6b2d19"

    @staticmethod
    def _write_min_config(text: str) -> None:
        paths = resolve_paths()
        os.makedirs(paths.data_dir, exist_ok=True)
        Path(paths.config).write_text(text, encoding="utf-8")

    def test_config_show_redacts_an_env_supplied_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An env-supplied api_key lands in the same SecretStr field a file value
        # would, so `config show` masks it exactly like a file secret.
        monkeypatch.setenv("PEARLARR_SONARR__API_KEY", self.ENV_KEY)
        self._write_min_config("sonarr:\n  url: http://127.0.0.1:9\n")
        result = CliRunner().invoke(pearlarr_cli, ["config", "show"])
        assert result.exit_code == 0
        assert "REDACTED" in result.output
        assert self.ENV_KEY not in result.output

    def test_config_validate_hides_env_secret_when_another_key_is_invalid(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The env supplies a valid secret while a bad file key drives the error
        # path; the error names the bad key and never echoes the env secret.
        monkeypatch.setenv("PEARLARR_SONARR__API_KEY", self.ENV_KEY)
        self._write_min_config("advanced:\n  sleep_time: not-a-number\n")
        result = CliRunner().invoke(pearlarr_cli, ["config", "validate"])
        assert result.exit_code == 1
        assert "sleep_time" in result.output
        assert self.ENV_KEY not in result.output


class TestArrClientMessages:
    """An arr connection error masks a url-embedded login while still naming the host."""

    def test_connection_error_masks_a_url_login_but_names_the_host(self) -> None:
        # A user:pass@ login in the arr URL is real basic auth (a protected
        # reverse proxy); the could-not-reach error keeps the host, never the login.
        url = f"http://admin:{CANARIES['url-embedded login']}@sonarr.local:8989"
        with respx.mock:
            respx.route().mock(side_effect=httpx.ConnectError("blocked"))
            arr = ArrHttp.bind(client=httpx.Client(), url=url, api_key="k", label="Sonarr", sleep=lambda _: None)
            with pytest.raises(ArrConnectionError) as excinfo:
                arr.get_json_list_strict("/api/v3/series")
        message = str(excinfo.value)
        _assert_clean(message)
        assert "REDACTED@sonarr.local:8989" in message


class TestNotifierFailureWarnings:
    """A Discord/webhook push failure warns by naming the config key, never the url (with its embedded token)."""

    @staticmethod
    def _result() -> WaitResult:
        return WaitResult(rows=(WaitOutcomeRow(label="Frieren S01", outcome=Outcome.IMPORTED),), elapsed_s=12.5)

    def test_discord_failure_warning_names_the_key_not_the_url(self) -> None:
        recording = install_recording_hub()
        with respx.mock:
            respx.post(DISCORD_URL).respond(500)
            notifier = Notifier(discord_url=DISCORD_URL, web=httpx.Client())
            assert notifier.push_wait_summary(arr=Arr.SONARR, result=self._result()) is False
        warnings = diagnostic_messages(recording)
        assert any("notifications.discord_url" in message for message in warnings)
        _assert_clean("\n".join(warnings))

    def test_webhook_failure_warning_names_the_key_not_the_url(self) -> None:
        recording = install_recording_hub()
        with respx.mock:
            respx.post(WEBHOOK_URL).mock(side_effect=httpx.ConnectError("boom"))
            notifier = Notifier(discord_url=None, webhook_url=WEBHOOK_URL, web=httpx.Client())
            assert notifier.push_wait_summary(arr=Arr.SONARR, result=self._result()) is False
        warnings = diagnostic_messages(recording)
        assert any("notifications.wait_webhook_url" in message for message in warnings)
        _assert_clean("\n".join(warnings))
