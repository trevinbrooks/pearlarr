# pyright: strict
"""Characterization tests for the Pydantic ``AppConfig`` model tree.

Pins the validated-settings behaviour: per-group defaults, blank-YAML coalescing,
strict validation (extra-forbid + strict enum), the lazy point-of-use connection
requirement, and the file lifecycle (template copy + checksum).
"""

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from seadexarr.modules.config import (
    PRIVATE_TRACKERS,
    PUBLIC_TRACKERS,
    AdvancedSettings,
    AppConfig,
    Arr,
    ImportsSettings,
    NotificationsSettings,
    PrivateReleaseAction,
    QbittorrentSettings,
    SeadexSettings,
)
from seadexarr.modules.manual_import import ImportWaitMode


class TestFileLifecycle:
    def test_missing_config_copies_template_and_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))
        # The bundled (nested) template was copied into place for the user to edit.
        assert cfg_path.exists()
        assert "sonarr" in yaml.safe_load(cfg_path.read_text())

    def test_load_preserves_user_values(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr:\n  url: http://x\nseadex:\n  private_releases: allow\n")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.sonarr.url == "http://x"
        assert cfg.seadex.private_releases is PrivateReleaseAction.ALLOW
        assert cfg.seadex.public_only is False

    def test_template_loads_as_all_defaults(self, tmp_path: Path) -> None:
        # The shipped template is all-blank, so it must validate to the defaults.
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))  # copies the template
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN
        assert cfg.imports.wait_mode is ImportWaitMode.OFF
        assert cfg.advanced.log_level == "INFO"

    def test_checksum_matches_file_bytes(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_bytes(b"seadex:\n  private_releases: warn\n")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.checksum() == hashlib.md5(b"seadex:\n  private_releases: warn\n").hexdigest()

    def test_load_parses_yaml_quirk_values_end_to_end(self, tmp_path: Path) -> None:
        # Guards the real file -> yaml.safe_load -> AppConfig.load path (not just
        # model_validate) for the three parse-quirk coalescings: an unquoted `off`
        # (which YAML 1.1 parses as the bool False), a scalar tracker, and an explicit
        # empty language list - all under their nested groups.
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(
            "seadex:\n  trackers: Nyaa\nimports:\n  wait_mode: off\n  languages_dual: []\n",
        )
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.seadex.trackers == {"nyaa"}
        assert cfg.imports.wait_mode is ImportWaitMode.OFF
        assert cfg.imports.languages_dual == ["Japanese", "English"]


class TestDefaults:
    def test_group_defaults_when_absent(self) -> None:
        cfg = AppConfig()
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN
        # The derived predicate: WARN and FALLBACK both mean "never grab private".
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
        assert cfg.advanced.detect_arr_activity is True
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
        assert imp.progress_poll_interval == 5
        assert imp.mode == "auto"
        assert imp.default_quality is None
        assert imp.languages_dual == ["Japanese", "English"]
        assert imp.languages_single == ["Japanese"]
        assert imp.pending_max_age_days == 14
        assert imp.digest_interval == 300

    def test_progress_poll_interval_zero_disables_the_fast_poll(self) -> None:
        # 0 is the documented "disable the cheap progress poll" value.
        assert ImportsSettings.model_validate({"progress_poll_interval": 0}).progress_poll_interval == 0
        assert ImportsSettings.model_validate({"progress_poll_interval": 3}).progress_poll_interval == 3

    def test_nested_explicit_values_override(self) -> None:
        cfg = AppConfig.model_validate(
            {
                "seadex": {"private_releases": "allow", "want_best": False},
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

    def test_explicit_empty_language_lists_fall_back_to_defaults(self) -> None:
        # An explicit `[]` (not just blank/None) must also coalesce to the default: the
        # pre-Pydantic property used truthiness (`value if value else default`), and an
        # imported file must never be tagged with no language (Sonarr reads that as
        # Unknown and may re-grab).
        imp = ImportsSettings.model_validate({"languages_dual": [], "languages_single": []})
        assert imp.languages_dual == ["Japanese", "English"]
        assert imp.languages_single == ["Japanese"]

    def test_blank_top_level_group_uses_defaults(self) -> None:
        # A group header with nothing under it (`seadex:`) parses to None.
        cfg = AppConfig.model_validate({"seadex": None})
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN


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

    def test_private_releases_coerces_valid_strings(self) -> None:
        for raw, member in [
            ("allow", PrivateReleaseAction.ALLOW),
            ("warn", PrivateReleaseAction.WARN),
            ("fallback", PrivateReleaseAction.FALLBACK),
        ]:
            assert SeadexSettings.model_validate({"private_releases": raw}).private_releases is member

    def test_private_releases_invalid_raises(self) -> None:
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"private_releases": "maybe"})

    def test_removed_public_only_key_is_rejected(self) -> None:
        # public_only was folded into private_releases (allow/warn/fallback); an
        # old config still setting it must fail loudly, not be silently ignored.
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"public_only": True})

    def test_wait_mode_coerces_valid_strings(self) -> None:
        assert ImportsSettings.model_validate({"wait_mode": "hybrid"}).wait_mode is ImportWaitMode.HYBRID
        assert ImportsSettings.model_validate({"wait_mode": "deferred"}).wait_mode is ImportWaitMode.DEFERRED
        assert ImportsSettings.model_validate({"wait_mode": "blocking"}).wait_mode is ImportWaitMode.BLOCKING
        assert ImportsSettings.model_validate({"wait_mode": "off"}).wait_mode is ImportWaitMode.OFF

    def test_wait_mode_invalid_raises(self) -> None:
        # Strict: an unrecognized wait_mode is a ValidationError (no lenient fallback).
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"wait_mode": "bogus"})

    def test_import_mode_accepts_documented_values(self) -> None:
        assert ImportsSettings.model_validate({"mode": "move"}).mode == "move"
        assert ImportsSettings.model_validate({"mode": "copy"}).mode == "copy"

    def test_import_mode_invalid_raises(self) -> None:
        # Strict: a typo'd importMode is a ValidationError at load, not a Sonarr API error.
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"mode": "moove"})

    def test_wait_mode_unquoted_yaml_off_maps_to_off(self) -> None:
        # YAML 1.1 parses a bare `off` (the documented disabled value) as the bool
        # False; it must still resolve to OFF rather than failing enum validation and
        # skipping the whole run.
        parsed = yaml.safe_load("wait_mode: off")
        assert parsed == {"wait_mode": False}
        assert ImportsSettings.model_validate(parsed).wait_mode is ImportWaitMode.OFF

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

    def test_trackers_explicit_empty_coalesces_to_default(self) -> None:
        # An explicit `[]` (or "") must coalesce to all trackers, not match nothing -
        # mirroring the languages validator. Empty means "no restriction", never a
        # silent grab-nothing. (A bare `trackers:` -> None is dropped to the default.)
        assert SeadexSettings.model_validate({"trackers": []}).trackers == PUBLIC_TRACKERS | PRIVATE_TRACKERS
        assert SeadexSettings.model_validate({"trackers": ""}).trackers == PUBLIC_TRACKERS | PRIVATE_TRACKERS

    def test_trackers_explicit_are_casefolded(self) -> None:
        assert SeadexSettings.model_validate({"trackers": ["Nyaa", "AB"]}).trackers == {"nyaa", "ab"}

    def test_trackers_scalar_string_is_single_tracker(self) -> None:
        # `trackers: Nyaa` (a natural scalar mistake) is one tracker, not iterated
        # character-by-character into a garbage {'n', 'y', 'a'} set that matches nothing.
        assert SeadexSettings.model_validate({"trackers": "Nyaa"}).trackers == {"nyaa"}

    def test_trackers_non_iterable_raises_validation_error(self) -> None:
        # A non-iterable must surface as a clean ValidationError (which the cli catches
        # and reports), not a raw TypeError that escapes to the generic handler.
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"trackers": 5})

    def test_log_level_is_uppercased(self) -> None:
        assert AdvancedSettings.model_validate({"log_level": "debug"}).log_level == "DEBUG"

    def test_log_level_error_is_accepted(self) -> None:
        # ERROR is a real level now (log.py honors it); it must validate like the rest.
        assert AdvancedSettings.model_validate({"log_level": "error"}).log_level == "ERROR"

    def test_log_level_typo_raises_validation_error(self) -> None:
        # Constrained at load: a typo is a clean ValidationError, not the logger's
        # runtime warn-and-default (which stays for non-config setup_logger callers).
        with pytest.raises(ValidationError):
            AdvancedSettings.model_validate({"log_level": "VERBOSE"})


class TestQbittorrent:
    def test_credentials_require_all_three(self) -> None:
        info = {"host": "h", "username": "u", "password": "p"}
        assert QbittorrentSettings.model_validate(info).credentials() == ("h", "u", "p")

    def test_credentials_none_when_any_missing(self) -> None:
        assert QbittorrentSettings.model_validate({"host": "h", "username": "u"}).credentials() is None
        assert QbittorrentSettings().credentials() is None

    def test_options_escape_hatch_passthrough_and_default(self) -> None:
        # `options` carries extra qbittorrentapi.Client kwargs (e.g. VERIFY_WEBUI_-
        # CERTIFICATE) that the explicit 4-field model would otherwise drop.
        assert QbittorrentSettings().options == {}
        cfg = QbittorrentSettings.model_validate(
            {"host": "h", "options": {"VERIFY_WEBUI_CERTIFICATE": False}},
        )
        assert cfg.options == {"VERIFY_WEBUI_CERTIFICATE": False}


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
