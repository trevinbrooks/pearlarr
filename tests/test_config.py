"""Characterization tests for the Pydantic ``AppConfig`` model tree.

Pins the validated-settings behaviour: per-group defaults, blank-YAML coalescing,
strict validation (extra-forbid + strict enum), the lazy point-of-use connection
requirement, and the file lifecycle (template copy + checksum).
"""

import hashlib

import pytest
import yaml
from pydantic import ValidationError

from seadexarr.modules.config import (
    PRIVATE_TRACKERS,
    PUBLIC_TRACKERS,
    AppConfig,
    Arr,
    ImportsSettings,
    NotificationsSettings,
    QbittorrentSettings,
    SeadexSettings,
)
from seadexarr.modules.manual_import import ImportWaitMode


class TestFileLifecycle:
    def test_missing_config_copies_template_and_raises(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))
        # The bundled (nested) template was copied into place for the user to edit.
        assert cfg_path.exists()
        assert "sonarr" in yaml.safe_load(cfg_path.read_text())

    def test_load_preserves_user_values(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr:\n  url: http://x\nseadex:\n  public_only: false\n")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.sonarr.url == "http://x"
        assert cfg.seadex.public_only is False

    def test_template_loads_as_all_defaults(self, tmp_path) -> None:
        # The shipped template is all-blank, so it must validate to the defaults.
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))  # copies the template
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.seadex.public_only is True
        assert cfg.imports.wait_mode is ImportWaitMode.OFF
        assert cfg.advanced.log_level == "INFO"

    def test_checksum_matches_file_bytes(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_bytes(b"seadex:\n  public_only: true\n")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.checksum() == hashlib.md5(b"seadex:\n  public_only: true\n").hexdigest()


class TestDefaults:
    def test_group_defaults_when_absent(self) -> None:
        cfg = AppConfig()
        assert cfg.seadex.public_only is True
        assert cfg.seadex.prefer_dual_audio is True
        assert cfg.seadex.want_best is True
        assert cfg.seadex.ignore_seadex_update_times is False
        assert cfg.seadex.use_torrent_hash_to_filter is False
        assert cfg.advanced.interactive is False
        assert cfg.advanced.sleep_time == 2
        assert cfg.advanced.cache_time == 1
        assert cfg.advanced.log_level == "INFO"
        assert cfg.advanced.max_torrents_to_add is None
        assert cfg.notifications.discord_url is None
        assert cfg.qbittorrent.tags is None
        assert cfg.qbittorrent.credentials() is None
        assert cfg.sonarr.torrent_category is None
        assert cfg.mappings.anime_mappings is None

    def test_import_defaults_when_absent(self) -> None:
        imp = ImportsSettings()
        assert imp.wait_mode is ImportWaitMode.OFF
        assert imp.wait_timeout == 3600
        assert imp.ready_timeout == 600
        assert imp.poll_interval == 30
        assert imp.mode == "auto"
        assert imp.default_quality is None
        assert imp.languages_dual == ["Japanese", "English"]
        assert imp.languages_single == ["Japanese"]
        assert imp.pending_max_age_days == 14
        assert imp.digest_interval == 300

    def test_nested_explicit_values_override(self) -> None:
        cfg = AppConfig.model_validate(
            {
                "seadex": {"public_only": False, "want_best": False},
                "advanced": {"sleep_time": 9},
                "notifications": {"discord_url": "u"},
            },
        )
        assert cfg.seadex.public_only is False
        assert cfg.seadex.want_best is False
        assert cfg.advanced.sleep_time == 9
        assert cfg.notifications.discord_url == "u"


class TestBlankCoalescing:
    def test_blank_scalar_knobs_fall_back_to_defaults(self) -> None:
        # A present-but-blank YAML value parses to None; without coalescing it would
        # flow into time.sleep(None) / be rejected by the int field.
        imp = ImportsSettings.model_validate(
            {
                "wait_timeout": None,
                "poll_interval": None,
                "ready_timeout": None,
                "mode": None,
                "pending_max_age_days": None,
            },
        )
        assert imp.wait_timeout == 3600
        assert imp.poll_interval == 30
        assert imp.ready_timeout == 600
        assert imp.mode == "auto"
        assert imp.pending_max_age_days == 14

    def test_blank_language_lists_fall_back_to_defaults(self) -> None:
        imp = ImportsSettings.model_validate({"languages_dual": None, "languages_single": None})
        assert imp.languages_dual == ["Japanese", "English"]
        assert imp.languages_single == ["Japanese"]

    def test_blank_top_level_group_uses_defaults(self) -> None:
        # A group header with nothing under it (`seadex:`) parses to None.
        cfg = AppConfig.model_validate({"seadex": None})
        assert cfg.seadex.public_only is True


class TestStrictValidation:
    def test_unknown_key_with_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"public_onlyy": True})

    def test_unknown_key_blank_is_also_rejected(self) -> None:
        # Blank typos are caught too: _drop_blank_none keeps unknown keys regardless
        # of blankness, so extra="forbid" still flags them.
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"public_onlyy": None})

    def test_unknown_top_level_group_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"sonar": {"url": "x"}})

    def test_radarr_group_rejects_sonarr_only_key(self) -> None:
        # ignore_movies_in_radarr lives on SonarrSettings only.
        AppConfig.model_validate({"sonarr": {"ignore_movies_in_radarr": True}})
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"radarr": {"ignore_movies_in_radarr": True}})

    def test_bad_scalar_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"advanced": {"sleep_time": "abc"}})

    def test_wait_mode_coerces_valid_strings(self) -> None:
        assert ImportsSettings.model_validate({"wait_mode": "hybrid"}).wait_mode is ImportWaitMode.HYBRID
        assert ImportsSettings.model_validate({"wait_mode": "deferred"}).wait_mode is ImportWaitMode.DEFERRED
        assert ImportsSettings.model_validate({"wait_mode": "blocking"}).wait_mode is ImportWaitMode.BLOCKING
        assert ImportsSettings.model_validate({"wait_mode": "off"}).wait_mode is ImportWaitMode.OFF

    def test_wait_mode_invalid_raises(self) -> None:
        # Strict: an unrecognized wait_mode is a ValidationError (no lenient fallback).
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"wait_mode": "bogus"})

    def test_mapping_true_is_rejected(self) -> None:
        # Only false (disable) / blank (auto-download) / inline dict are valid.
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"mappings": {"anime_mappings": True}})
        assert AppConfig.model_validate({"mappings": {"anime_mappings": False}}).mappings.anime_mappings is False


class TestNormalization:
    def test_ignore_tags_none_becomes_empty_list(self) -> None:
        assert SeadexSettings().ignore_tags == []
        assert SeadexSettings.model_validate({"ignore_tags": None}).ignore_tags == []
        assert SeadexSettings.model_validate({"ignore_tags": ["a", "b"]}).ignore_tags == ["a", "b"]

    def test_ignore_anilist_ids_coerced_to_int_set(self) -> None:
        assert SeadexSettings().ignore_anilist_ids == set()
        assert SeadexSettings.model_validate({"ignore_anilist_ids": None}).ignore_anilist_ids == set()
        assert SeadexSettings.model_validate({"ignore_anilist_ids": ["1", "2", 3]}).ignore_anilist_ids == {1, 2, 3}

    def test_trackers_default_is_public_plus_private(self) -> None:
        assert SeadexSettings().trackers == PUBLIC_TRACKERS | PRIVATE_TRACKERS

    def test_trackers_explicit_are_casefolded(self) -> None:
        assert SeadexSettings.model_validate({"trackers": ["Nyaa", "AB"]}).trackers == {"nyaa", "ab"}


class TestQbittorrent:
    def test_credentials_require_all_three(self) -> None:
        info = {"host": "h", "username": "u", "password": "p"}
        assert QbittorrentSettings.model_validate(info).credentials() == ("h", "u", "p")

    def test_credentials_none_when_any_missing(self) -> None:
        assert QbittorrentSettings.model_validate({"host": "h", "username": "u"}).credentials() is None
        assert QbittorrentSettings().credentials() is None


class TestNotifications:
    def test_wait_notify_defaults_off_without_webhooks(self) -> None:
        assert NotificationsSettings().wait_notify is False

    def test_wait_notify_on_when_any_webhook_set(self) -> None:
        assert NotificationsSettings.model_validate({"discord_url": "u"}).wait_notify is True
        assert NotificationsSettings.model_validate({"wait_webhook_url": "u"}).wait_notify is True

    def test_explicit_wait_notify_wins(self) -> None:
        assert NotificationsSettings.model_validate({"discord_url": "u", "wait_notify": False}).wait_notify is False


class TestConnection:
    def test_for_arr_selects_submodel(self) -> None:
        cfg = AppConfig.model_validate(
            {
                "sonarr": {"ignore_unmonitored": True, "torrent_category": "anime-tv"},
                "radarr": {"ignore_unmonitored": False, "torrent_category": "anime-movies"},
            },
        )
        assert cfg.for_arr(Arr.SONARR).ignore_unmonitored is True
        assert cfg.for_arr(Arr.RADARR).ignore_unmonitored is False
        assert cfg.for_arr(Arr.SONARR).torrent_category == "anime-tv"
        assert cfg.for_arr(Arr.RADARR).torrent_category == "anime-movies"

    def test_require_connection_returns_url_api_key(self) -> None:
        cfg = AppConfig.model_validate(
            {
                "sonarr": {"url": "http://s", "api_key": "sk"},
                "radarr": {"url": "http://r", "api_key": "rk"},
            },
        )
        assert cfg.require_connection(Arr.SONARR) == ("http://s", "sk")
        assert cfg.require_connection(Arr.RADARR) == ("http://r", "rk")

    def test_require_connection_raises_when_absent(self) -> None:
        cfg = AppConfig()
        for arr in (Arr.SONARR, Arr.RADARR):
            with pytest.raises(ValueError):
                cfg.require_connection(arr)

    def test_cross_check_reads_are_none_tolerant(self) -> None:
        # The Sonarr->Radarr specials cross-check reads radarr url/api_key directly
        # (no require), so they stay None on a Sonarr-only config.
        cfg = AppConfig.model_validate({"sonarr": {"url": "http://s", "api_key": "sk"}})
        assert cfg.radarr.url is None
        assert cfg.radarr.api_key is None
