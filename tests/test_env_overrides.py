# pyright: strict
# pyright: reportPrivateUsage=false
# The overlay unit tests read the private `_env_overlay`/`_hash_config_bytes` to pin
# the nested-mapping shape and the digest formula directly (both unobservable
# through a validated AppConfig). Strict re-flags that and the repo disables
# reportPrivateUsage for tests.
"""Environment-variable config overrides: the `PEARLARR_<GROUP>__<KEY>` overlay.

Pins the overlay's path mapping and YAML value parsing, the leaf-wins/
dicts-merge semantics against a real file, the reserved delimiter-less names,
the error line naming the variable that set a bad key, and the checksum folding
a qualifying env change in.
"""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from pearlarr.bootstrap import format_validation_errors
from pearlarr.config import AppConfig, _env_overlay, _hash_config_bytes
from pearlarr.wait_types import ImportWaitMode


class TestOverlayShape:
    """`_env_overlay` maps a qualifying name to a key path with a YAML value."""

    def test_path_mapping_and_lowercasing(self) -> None:
        assert _env_overlay({"PEARLARR_SONARR__URL": "http://x"}) == {"sonarr": {"url": "http://x"}}

    def test_only_delimiter_segments_split_underscores_within_a_key_survive(self) -> None:
        # A single underscore is part of the key. Only `__` nests.
        assert _env_overlay({"PEARLARR_SEADEX__WANT_BEST": "false"}) == {"seadex": {"want_best": False}}

    def test_deeper_paths_nest_preserving_free_form_key_case(self) -> None:
        # Keys below qbittorrent.options reach qbittorrentapi verbatim and
        # case-sensitively. Folding them would make VERIFY_WEBUI_CERTIFICATE unreachable.
        assert _env_overlay({"PEARLARR_QBITTORRENT__OPTIONS__VERIFY_WEBUI_CERTIFICATE": "false"}) == {
            "qbittorrent": {"options": {"VERIFY_WEBUI_CERTIFICATE": False}},
        }

    def test_segments_below_a_free_form_table_stay_verbatim_all_the_way_down(self) -> None:
        assert _env_overlay({"PEARLARR_QBITTORRENT__OPTIONS__REQUESTS_ARGS__timeout": "5"}) == {
            "qbittorrent": {"options": {"REQUESTS_ARGS": {"timeout": 5}}},
        }

    def test_two_vars_share_a_group(self) -> None:
        overlay = _env_overlay({"PEARLARR_SONARR__URL": "http://x", "PEARLARR_SONARR__API_KEY": "k"})
        assert overlay == {"sonarr": {"url": "http://x", "api_key": "k"}}

    def test_delimiter_less_names_are_reserved(self) -> None:
        # The operational vars (data dir, Docker) carry no `__` and must never be
        # swallowed into config. A non-prefixed name is ignored too.
        overlay = _env_overlay(
            {
                "PEARLARR_DATA_DIR": "/data",
                "PEARLARR_CRON": "* * * * *",
                "PEARLARR_RUN_ON_START": "true",
                "OTHER": "x",
            },
        )
        assert overlay == {}


class TestOverlayValueParsing:
    """A value is `yaml.safe_load`ed, so it means exactly what the same text means in the file."""

    def test_int_bool_and_flow_list(self) -> None:
        overlay = _env_overlay(
            {
                "PEARLARR_ADVANCED__SLEEP_TIME": "0",
                "PEARLARR_SONARR__VERIFY_SSL": "false",
                "PEARLARR_SEADEX__IGNORE_TAGS": "[Dolby Vision, Deband Required]",
            },
        )
        assert overlay["advanced"] == {"sleep_time": 0}
        assert overlay["sonarr"] == {"verify_ssl": False}
        assert overlay["seadex"] == {"ignore_tags": ["Dolby Vision", "Deband Required"]}

    def test_quotes_force_a_string(self) -> None:
        # An API key that reads as a number needs quoting, exactly as in the file.
        assert _env_overlay({"PEARLARR_SONARR__API_KEY": "'123'"}) == {"sonarr": {"api_key": "123"}}

    def test_empty_value_parses_to_none(self) -> None:
        assert _env_overlay({"PEARLARR_SONARR__URL": ""}) == {"sonarr": {"url": None}}

    def test_malformed_yaml_names_only_the_variable(self) -> None:
        with pytest.raises(ValidationError, match=r"PEARLARR_SONARR__API_KEY") as excinfo:
            _env_overlay({"PEARLARR_SONARR__API_KEY": "'unterminated"})
        assert "unterminated" not in str(excinfo.value)
        # The invalid-configuration arms render this like a bad file key.
        assert "PEARLARR_SONARR__API_KEY" in format_validation_errors(excinfo.value)


class TestOverlayThroughLoad:
    """`AppConfig.load` overlays the environment onto the migrated file, env winning per leaf."""

    def _write(self, tmp_path: Path, text: str) -> str:
        cfg = tmp_path / "config.yml"
        cfg.write_text(text, encoding="utf-8")
        return str(cfg)

    def test_leaf_override_keeps_the_file_sibling(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = self._write(tmp_path, "sonarr:\n  url: http://file\n  api_key: filekey\n")
        monkeypatch.setenv("PEARLARR_SONARR__URL", "http://env")
        loaded = AppConfig.load(path)
        assert loaded.sonarr.url == "http://env"
        assert loaded.sonarr.api_key is not None
        assert loaded.sonarr.api_key.get_secret_value() == "filekey"

    def test_env_beats_the_file_and_parses_the_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = self._write(tmp_path, "imports:\n  wait_mode: off\nadvanced:\n  sleep_time: 9\n")
        monkeypatch.setenv("PEARLARR_IMPORTS__WAIT_MODE", "hybrid")
        monkeypatch.setenv("PEARLARR_ADVANCED__SLEEP_TIME", "0")
        loaded = AppConfig.load(path)
        assert loaded.imports.wait_mode is ImportWaitMode.HYBRID
        assert loaded.advanced.sleep_time == 0

    def test_free_form_option_key_lands_case_intact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The docs' canonical deeper-path example, end to end through validation.
        path = self._write(tmp_path, "sonarr:\n  url: http://x\n  api_key: k\n")
        monkeypatch.setenv("PEARLARR_QBITTORRENT__OPTIONS__VERIFY_WEBUI_CERTIFICATE", "false")
        assert AppConfig.load(path).qbittorrent.options == {"VERIFY_WEBUI_CERTIFICATE": False}

    def test_blank_value_falls_back_to_the_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # An empty env value reads like a blank key in the file: None, then the
        # blank-handling coalesces it to the built-in default.
        path = self._write(tmp_path, "advanced:\n  sleep_time: 9\n")
        monkeypatch.setenv("PEARLARR_ADVANCED__SLEEP_TIME", "")
        assert AppConfig.load(path).advanced.sleep_time == 0

    def test_non_mapping_file_skips_the_merge(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A junk (non-mapping) file lets its own validation error stand unchanged.
        path = self._write(tmp_path, "- just\n- a\n- list\n")
        monkeypatch.setenv("PEARLARR_SONARR__URL", "http://x")
        with pytest.raises(ValidationError):
            AppConfig.load(path)


class TestErrorLinesNameTheVariable:
    """A config error line names the environment variable that set its key."""

    def _error(self, tmp_path: Path, text: str) -> str:
        cfg = tmp_path / "config.yml"
        cfg.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as excinfo:
            AppConfig.load(str(cfg))
        return format_validation_errors(excinfo.value)

    def test_unknown_key_names_its_variable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_SEADEX__WANT_BST", "false")
        assert self._error(tmp_path, "") == "  - seadex.want_bst (set by PEARLARR_SEADEX__WANT_BST): unknown key"

    def test_bad_value_names_its_variable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_ADVANCED__SLEEP_TIME", "soon")
        error = self._error(tmp_path, "")
        assert error.startswith("  - advanced.sleep_time (set by PEARLARR_ADVANCED__SLEEP_TIME): Input should be")

    def test_unknown_group_names_every_variable_under_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PEARLARR_SONAR__VERIFY_SSL", "false")
        monkeypatch.setenv("PEARLARR_SONAR__LANGUAGE", "English")
        error = self._error(tmp_path, "")
        assert error == "  - sonar (set by PEARLARR_SONAR__LANGUAGE, PEARLARR_SONAR__VERIFY_SSL): unknown key"

    def test_bad_list_item_names_the_variable_that_set_the_list(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PEARLARR_SEADEX__IGNORE_ANILIST_IDS", "[1, nope]")
        error = self._error(tmp_path, "")
        assert error.startswith("  - seadex.ignore_anilist_ids.1 (set by PEARLARR_SEADEX__IGNORE_ANILIST_IDS): ")

    def test_file_key_names_no_variable_of_its_group(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_ADVANCED__LOG_LEVEL", "INFO")
        assert self._error(tmp_path, "advanced:\n  sleep_tme: 5\n") == "  - advanced.sleep_tme: unknown key"

    def test_non_mapping_file_names_no_variable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_ADVANCED__LOG_LEVEL", "INFO")
        assert self._error(tmp_path, "- just\n- a\n- list\n").startswith("  - : Input should be")

    def test_unparseable_value_names_its_variable_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PEARLARR_ADVANCED__SLEEP_TIME", "'unterminated")
        error = self._error(tmp_path, "")
        assert error == "  - PEARLARR_ADVANCED__SLEEP_TIME: is not valid YAML - fix or unset the environment variable"


class TestChecksumFoldsEnv:
    """The cache descriptor folds the qualifying env pairs in, so an env change re-checks."""

    def test_digest_folds_only_qualifying_pairs(self) -> None:
        raw = b"sonarr:\n  url: http://x\n"
        environ = {"PEARLARR_SEADEX__WANT_BEST": "false", "PEARLARR_DATA_DIR": "/d", "OTHER": "y"}
        assert (
            _hash_config_bytes(raw, environ)
            == hashlib.md5(
                raw + b"PEARLARR_SEADEX__WANT_BEST=false\n",
            ).hexdigest()
        )
        # Unrelated (delimiter-less / non-prefixed) vars leave it at the no-env digest.
        assert _hash_config_bytes(raw, {"PEARLARR_DATA_DIR": "/d", "OTHER": "y"}) == hashlib.md5(raw).hexdigest()

    def test_checksum_moves_when_an_override_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_bytes(b"sonarr:\n  url: http://x\n")
        base = AppConfig.load(str(cfg)).config_hash()
        monkeypatch.setenv("PEARLARR_SEADEX__WANT_BEST", "false")
        assert AppConfig.load(str(cfg)).config_hash() != base

    def test_checksum_stable_for_unrelated_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_bytes(b"sonarr:\n  url: http://x\n")
        base = AppConfig.load(str(cfg)).config_hash()
        monkeypatch.setenv("PEARLARR_DATA_DIR", str(tmp_path / "elsewhere"))
        monkeypatch.setenv("UNRELATED", "x")
        assert AppConfig.load(str(cfg)).config_hash() == base
