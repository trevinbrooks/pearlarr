# pyright: strict
"""Characterization tests for the Pydantic `AppConfig` model tree.

Pins the validated-settings behavior: per-group defaults, blank-YAML coalescing,
strict validation (extra-forbid + strict enum), the lazy point-of-use connection
requirement, and the file lifecycle (template copy + checksum).
"""

import hashlib
import os
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr, ValidationError
from seadex import Tracker

from pearlarr.modules.config import (
    KNOWN_TRACKERS,
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
    config_permissions_loose,
    secret_value,
)
from pearlarr.modules.config_migrations import CONFIG_VERSION, MigrationOutcome
from pearlarr.modules.manual_import import ImportWaitMode


class TestFileLifecycle:
    """A missing config copies the starter template (0600, banner-stripped) and raises.

    An existing one loads normally, preserving user values, with a byte-exact checksum.
    """

    def test_missing_config_copies_template_and_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        # The message reflects that the copy already happened and says what to
        # do next (standalone callers see it raw; the CLI logs its own version).
        with pytest.raises(FileNotFoundError, match="starter template was written"):
            AppConfig.load(str(cfg_path))
        # The bundled (nested) template was copied into place for the user to edit.
        assert cfg_path.exists()
        assert "sonarr" in yaml.safe_load(cfg_path.read_text())

    def test_starter_copy_drops_generated_banner(self, tmp_path: Path) -> None:
        # The template is a generated artifact; the copy is the user's file to
        # edit, so the banner goes while the editor-schema pointer survives.
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))
        text = cfg_path.read_text()
        assert "GENERATED" not in text
        assert "$schema=" in text

    @pytest.mark.skipif(os.name != "posix", reason="mode bits are POSIX-only")
    def test_template_copy_is_owner_only(self, tmp_path: Path) -> None:
        # The config holds plaintext API keys: the first-run template copy must
        # land 0600, not inherit the world-readable mode of the bundled template.
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path))
        assert (cfg_path.stat().st_mode & 0o777) == 0o600

    @pytest.mark.skipif(os.name != "posix", reason="mode bits are POSIX-only")
    def test_loose_permissions_detected_and_600_accepted(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr:\n  url: http://x\n")
        cfg_path.chmod(0o644)
        assert config_permissions_loose(str(cfg_path)) is True
        cfg_path.chmod(0o600)
        assert config_permissions_loose(str(cfg_path)) is False
        # An unstatable path is not "loose" (the load path reports missing files itself).
        assert config_permissions_loose(str(tmp_path / "absent.yml")) is False

    def test_load_preserves_user_values(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr:\n  url: http://x\nseadex:\n  private_releases: fallback\n")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg.sonarr.url == "http://x"
        assert cfg.seadex.private_releases is PrivateReleaseAction.FALLBACK

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


class TestSchemaMigration:
    """`load` brings an older file's mapping to the current schema, in memory only.

    A file without `config_version` is pre-versioning (v0): its removed keys and
    values are folded forward, the outcome is reported via `migration()`, and
    the file on disk stays byte-identical. A current file migrates nothing; a
    newer one is refused by name.
    """

    def _load(self, tmp_path: Path, text: str) -> AppConfig:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(text, encoding="utf-8")
        return AppConfig.load(str(cfg_path))

    def test_v0_file_with_every_delta_loads_and_reports_each(self, tmp_path: Path) -> None:
        cfg = self._load(
            tmp_path,
            "seadex:\n  public_only: true\nadvanced:\n  log_level: trace\nimports:\n  mode: hardlink\n",
        )
        # Each fold lands on the removed key's old runtime behavior.
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN
        assert cfg.advanced.log_level == "INFO"
        assert cfg.imports.mode == "auto"
        assert cfg.config_version == CONFIG_VERSION
        outcome = cfg.migration()
        assert outcome is not None
        assert outcome.from_version == 0
        assert len(outcome.notes) == 3
        assert any("public_only" in note for note in outcome.notes)

    def test_removed_allow_value_folds_to_warn(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path, "seadex:\n  private_releases: allow\n")
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN
        outcome = cfg.migration()
        assert outcome is not None
        assert any("'allow'" in note for note in outcome.notes)

    def test_v0_file_without_deltas_is_stamped_silently(self, tmp_path: Path) -> None:
        # Nothing to fold: the pass stamps the version and carries no notes.
        cfg = self._load(tmp_path, "sonarr:\n  url: http://x\n")
        assert cfg.sonarr.url == "http://x"
        outcome = cfg.migration()
        assert outcome is not None
        assert outcome == MigrationOutcome(from_version=0, notes=())

    def test_current_file_migrates_nothing(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path, f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://x\n")
        assert cfg.migration() is None

    def test_blank_version_counts_as_pre_versioning(self, tmp_path: Path) -> None:
        # `config_version:` parses to None; the blank-drop would default it, but
        # the loader must still treat the file as v0 and stamp it.
        cfg = self._load(tmp_path, "config_version:\n")
        outcome = cfg.migration()
        assert outcome is not None
        assert outcome.from_version == 0
        assert cfg.config_version == CONFIG_VERSION

    def test_newer_file_is_refused_naming_both_versions(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="newer Pearlarr"):
            self._load(tmp_path, f"config_version: {CONFIG_VERSION + 1}\n")

    def test_non_int_version_is_a_validation_error_not_a_migration(self, tmp_path: Path) -> None:
        # Garbage must surface under its key, not be silently re-stamped.
        with pytest.raises(ValidationError, match="config_version"):
            self._load(tmp_path, "config_version: banana\n")

    def test_migration_never_touches_the_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yml"
        text = "seadex:\n  private_releases: allow\n"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = AppConfig.load(str(cfg_path))
        assert cfg_path.read_text(encoding="utf-8") == text
        # The checksum (the cache descriptor) hashes the on-disk bytes, so an
        # in-memory migration cannot shift it between runs.
        assert cfg.checksum() == hashlib.md5(text.encode()).hexdigest()

    def test_direct_validation_keeps_the_version_floor(self) -> None:
        with pytest.raises(ValidationError, match="config_version"):
            AppConfig.model_validate({"config_version": 0})


class TestDefaults:
    """Every group defaults sensibly when absent, and numeric knobs enforce their documented bounds.

    An explicit nested value always overrides the default.
    """

    def test_group_defaults_when_absent(self) -> None:
        cfg = AppConfig()
        assert cfg.seadex.private_releases is PrivateReleaseAction.WARN
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
        assert cfg.schedule.interval_hours == 6.0
        assert cfg.notifications.discord_url is None
        assert cfg.qbittorrent.tags is None
        assert cfg.qbittorrent.credentials() is None
        assert cfg.sonarr.torrent_category is None
        assert cfg.sonarr.verify_ssl is True
        assert cfg.radarr.verify_ssl is True
        assert cfg.mappings.anime_mappings is None

    def test_verify_ssl_parses_per_arr(self) -> None:
        cfg = AppConfig.model_validate({"radarr": {"verify_ssl": False}})
        assert cfg.radarr.verify_ssl is False
        assert cfg.sonarr.verify_ssl is True

    def test_schedule_interval_bounds(self) -> None:
        assert AppConfig.model_validate({"schedule": {"interval_hours": 0.5}}).schedule.interval_hours == 0.5
        for bad in (0, -3, float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                AppConfig.model_validate({"schedule": {"interval_hours": bad}})

    def test_wait_timeout_bounds(self) -> None:
        # ge=1: a zero wait window is a degenerate busy-loop, not a disable.
        assert ImportsSettings.model_validate({"wait_timeout": 1}).wait_timeout == 1
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"wait_timeout": 0})

    def test_ready_timeout_bounds(self) -> None:
        assert ImportsSettings.model_validate({"ready_timeout": 1}).ready_timeout == 1
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"ready_timeout": 0})

    def test_poll_interval_bounds(self) -> None:
        assert ImportsSettings.model_validate({"poll_interval": 1}).poll_interval == 1
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"poll_interval": 0})

    def test_progress_poll_interval_bounds(self) -> None:
        # 0 stays valid (the documented disable, pinned separately); negatives reject.
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"progress_poll_interval": -1})

    def test_pending_max_age_days_bounds(self) -> None:
        # ge=1: 0 would expire every pending-import record immediately.
        assert ImportsSettings.model_validate({"pending_max_age_days": 1}).pending_max_age_days == 1
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"pending_max_age_days": 0})

    def test_digest_interval_bounds(self) -> None:
        assert ImportsSettings.model_validate({"digest_interval": 1}).digest_interval == 1
        with pytest.raises(ValidationError):
            ImportsSettings.model_validate({"digest_interval": 0})

    def test_sleep_time_bounds(self) -> None:
        # 0 disables rate limiting (valid); a negative would crash time.sleep mid-run.
        assert AdvancedSettings.model_validate({"sleep_time": 0}).sleep_time == 0
        with pytest.raises(ValidationError):
            AdvancedSettings.model_validate({"sleep_time": -1})

    def test_cache_time_bounds(self) -> None:
        assert AdvancedSettings.model_validate({"cache_time": 0}).cache_time == 0
        with pytest.raises(ValidationError):
            AdvancedSettings.model_validate({"cache_time": -1})

    def test_max_torrents_to_add_bounds(self) -> None:
        # A cap of 0 would silently grab nothing; None stays the unlimited default.
        assert AdvancedSettings.model_validate({"max_torrents_to_add": 1}).max_torrents_to_add == 1
        assert AdvancedSettings.model_validate({"max_torrents_to_add": None}).max_torrents_to_add is None
        with pytest.raises(ValidationError):
            AdvancedSettings.model_validate({"max_torrents_to_add": 0})

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
                "seadex": {"private_releases": "fallback", "want_best": False},
                "advanced": {"sleep_time": 9},
                "notifications": {"discord_url": "u"},
            },
        )
        assert cfg.seadex.private_releases is PrivateReleaseAction.FALLBACK
        assert cfg.seadex.want_best is False
        assert cfg.advanced.sleep_time == 9
        assert cfg.notifications.discord_url == SecretStr("u")


class TestBlankCoalescing:
    """A present-but-blank YAML value (None, or an explicit empty list/group) coalesces to its default."""

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
    """`extra="forbid"` and strict enums reject unknown/wrong-group keys and bad or invalid enum values.

    Only the documented strings coerce; a removed key stays rejected too.
    """

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
            ("warn", PrivateReleaseAction.WARN),
            ("fallback", PrivateReleaseAction.FALLBACK),
        ]:
            assert SeadexSettings.model_validate({"private_releases": raw}).private_releases is member

    def test_private_releases_invalid_raises(self) -> None:
        with pytest.raises(ValidationError):
            SeadexSettings.model_validate({"private_releases": "maybe"})

    def test_removed_public_only_key_is_rejected(self) -> None:
        # public_only was folded into private_releases (warn/fallback); an
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
    """Field normalizers coerce/casefold tags, ids, trackers; log_level/log_format case-fold and reject typos."""

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

    def test_known_trackers_matches_the_installed_seadex_enum(self) -> None:
        # KNOWN_TRACKERS is spelled as literals (config.py must not import seadex -
        # boot depends on it staying light); pin them to the real enum so a seadex
        # upgrade that adds/renames a tracker can't drift silently.
        assert KNOWN_TRACKERS == {tracker.value.casefold() for tracker in Tracker}

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

    def test_log_format_defaults_to_auto_and_is_lowercased(self) -> None:
        assert AdvancedSettings().log_format == "auto"
        assert AdvancedSettings.model_validate({"log_format": "JSON"}).log_format == "json"

    def test_log_format_typo_raises_validation_error(self) -> None:
        # Constrained at load like log_level: a typo'd renderer name is a clean
        # ValidationError, not a runtime fallback.
        with pytest.raises(ValidationError):
            AdvancedSettings.model_validate({"log_format": "fancy"})


class TestQbittorrent:
    """`credentials()` returns a triple only when host/username/password are all set.

    `options` passes extra qbittorrent-api kwargs through untouched.
    """

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
    """`wait_notify` defaults on the moment any webhook URL is set, unless the field is set explicitly."""

    def test_wait_notify_defaults_off_without_webhooks(self) -> None:
        assert NotificationsSettings().wait_notify is False

    def test_wait_notify_on_when_any_webhook_set(self) -> None:
        assert NotificationsSettings.model_validate({"discord_url": "u"}).wait_notify is True
        assert NotificationsSettings.model_validate({"wait_webhook_url": "u"}).wait_notify is True

    def test_explicit_wait_notify_wins(self) -> None:
        assert NotificationsSettings.model_validate({"discord_url": "u", "wait_notify": False}).wait_notify is False


class TestConnection:
    """`for_arr`/`require_connection` select the per-arr submodel and enforce a fully configured connection."""

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


class TestSecrets:
    """The credential fields are `SecretStr`: masked everywhere but their point of use."""

    _RAW = {
        "sonarr": {"url": "http://s", "api_key": "sonarr-key"},
        "radarr": {"api_key": "radarr-key"},
        "qbittorrent": {"host": "h", "username": "u", "password": "hunter2"},
        "notifications": {
            "discord_url": "https://discord.com/api/webhooks/1/tok",
            "wait_webhook_url": "https://hook.example/secret",
        },
    }

    def test_secret_fields_parse_to_secretstr(self) -> None:
        cfg = AppConfig.model_validate(self._RAW)
        assert cfg.sonarr.api_key == SecretStr("sonarr-key")
        assert cfg.radarr.api_key == SecretStr("radarr-key")
        assert cfg.qbittorrent.password == SecretStr("hunter2")
        assert cfg.notifications.discord_url == SecretStr("https://discord.com/api/webhooks/1/tok")
        assert cfg.notifications.wait_webhook_url == SecretStr("https://hook.example/secret")

    def test_repr_and_json_dump_mask_every_secret(self) -> None:
        # The whole point of SecretStr: an incidentally logged/dumped config
        # (repr in a traceback, model_dump in a bug report) can't leak.
        cfg = AppConfig.model_validate(self._RAW)
        for rendering in (repr(cfg), str(cfg), str(cfg.model_dump(mode="json"))):
            for secret in ("sonarr-key", "radarr-key", "hunter2", "webhooks/1/tok", "hook.example/secret"):
                assert secret not in rendering

    def test_validation_errors_hide_the_input_value(self) -> None:
        # hide_input_in_errors: a credential pasted under the wrong key (here a
        # str where an int belongs) must not be echoed back in the error text.
        with pytest.raises(ValidationError) as excinfo:
            AppConfig.model_validate({"advanced": {"sleep_time": "hunter2"}})
        assert "hunter2" not in str(excinfo.value)

    def test_points_of_use_unwrap_to_plain_strings(self) -> None:
        # require_connection / credentials() are the sanctioned unwrap points;
        # their callers (client construction, qbit login) get plain strings.
        cfg = AppConfig.model_validate(self._RAW)
        assert cfg.require_connection(Arr.SONARR) == ("http://s", "sonarr-key")
        assert cfg.qbittorrent.credentials() == ("h", "u", "hunter2")

    def test_secret_value_unwraps_optionals(self) -> None:
        assert secret_value(SecretStr("tok")) == "tok"
        assert secret_value(None) is None

    def test_empty_secret_reads_as_unset(self) -> None:
        # missing_arr_keys / credentials() gate on truthiness; an empty SecretStr
        # (a blank quoted value) must read as unset, like an empty plain string.
        cfg = AppConfig.model_validate({"sonarr": {"url": "http://s", "api_key": ""}})
        assert cfg.missing_arr_keys(Arr.SONARR) == ("sonarr.api_key",)
