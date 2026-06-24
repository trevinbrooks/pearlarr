"""Characterization tests for ``AppConfig``.

Pins the config-file lifecycle and the typed-settings normalization moved out of
``SeaDexArr.__init__`` / ``verify_config`` in Phase 2 (see ``REFACTOR_PLAN.md``).
"""

import hashlib

import pytest
import yaml

from seadexarr.modules.config import PRIVATE_TRACKERS, PUBLIC_TRACKERS, AppConfig, Arr
from seadexarr.modules.manual_import import ImportWaitMode


def _cfg(**data: object) -> AppConfig:
    """An ``AppConfig`` over an in-memory data dict (no file load)."""

    return AppConfig(path="unused.yml", arr=Arr.SONARR, data=dict(data))


class TestFileLifecycle:
    def test_missing_config_copies_template_and_raises(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path), Arr.SONARR)
        # The bundled template was copied into place for the user to edit.
        assert cfg_path.exists()
        assert "sonarr_url" in yaml.safe_load(cfg_path.read_text())

    def test_load_preserves_user_values(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr_url: http://x\npublic_only: false\n")
        cfg = AppConfig.load(str(cfg_path), Arr.SONARR)
        assert cfg.data["sonarr_url"] == "http://x"
        assert cfg.public_only is False

    def test_checksum_matches_file_bytes(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_bytes(b"public_only: true\n")
        cfg = AppConfig(path=str(cfg_path), arr=Arr.SONARR, data={})
        assert cfg.checksum() == hashlib.md5(b"public_only: true\n").hexdigest()


class TestTemplateSync:
    def test_partial_config_is_rewritten_in_template_order(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        # Keys present but out of template order -> triggers the rewrite.
        cfg_path.write_text("public_only: false\nsonarr_url: http://x\n")
        cfg = AppConfig.load(str(cfg_path), Arr.SONARR)
        # First template key leads, a previously-absent key is now present, and
        # the user's values survive the merge.
        assert next(iter(cfg.data)) == "sonarr_url"
        assert "log_level" in cfg.data
        assert cfg.public_only is False
        assert cfg.data["sonarr_url"] == "http://x"

    def test_template_order_config_not_rewritten(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path), Arr.SONARR)  # copies the template
        before = cfg_path.read_bytes()
        AppConfig.load(str(cfg_path), Arr.SONARR)  # already in template order
        assert cfg_path.read_bytes() == before


class TestTypedSettings:
    def test_defaults_when_absent(self) -> None:
        cfg = _cfg()
        assert cfg.public_only is True
        assert cfg.prefer_dual_audio is True
        assert cfg.want_best is True
        assert cfg.ignore_seadex_update_times is False
        assert cfg.use_torrent_hash_to_filter is False
        assert cfg.interactive is False
        assert cfg.sleep_time == 2
        assert cfg.cache_time == 1
        assert cfg.log_level == "INFO"
        assert cfg.discord_url is None
        assert cfg.max_torrents_to_add is None
        assert cfg.torrent_tags is None
        assert cfg.qbit_info is None

    def test_explicit_values_override(self) -> None:
        cfg = _cfg(public_only=False, want_best=False, sleep_time=9, discord_url="u")
        assert cfg.public_only is False
        assert cfg.want_best is False
        assert cfg.sleep_time == 9
        assert cfg.discord_url == "u"

    def test_arr_prefixed_keys_select_by_arr(self) -> None:
        data = {
            "sonarr_ignore_unmonitored": True,
            "radarr_ignore_unmonitored": False,
            "sonarr_torrent_category": "anime-tv",
            "radarr_torrent_category": "anime-movies",
        }
        sonarr = AppConfig(path="x", arr=Arr.SONARR, data=data)
        radarr = AppConfig(path="x", arr=Arr.RADARR, data=data)
        assert sonarr.ignore_unmonitored is True
        assert radarr.ignore_unmonitored is False
        assert sonarr.torrent_category == "anime-tv"
        assert radarr.torrent_category == "anime-movies"

    def test_ignore_tags_none_becomes_empty_list(self) -> None:
        assert _cfg().ignore_tags == []
        assert _cfg(ignore_tags=None).ignore_tags == []
        assert _cfg(ignore_tags=["a", "b"]).ignore_tags == ["a", "b"]

    def test_ignore_anilist_ids_coerced_to_int_set(self) -> None:
        assert _cfg().ignore_anilist_ids == set()
        assert _cfg(ignore_anilist_ids=None).ignore_anilist_ids == set()
        assert _cfg(ignore_anilist_ids=["1", "2", 3]).ignore_anilist_ids == {1, 2, 3}

    def test_trackers_default_is_public_plus_private(self) -> None:
        assert _cfg().trackers == PUBLIC_TRACKERS | PRIVATE_TRACKERS

    def test_trackers_explicit_are_casefolded(self) -> None:
        assert _cfg(trackers=["Nyaa", "AB"]).trackers == {"nyaa", "ab"}

    def test_qbit_info_passthrough(self) -> None:
        info = {"host": "h", "username": "u", "password": "p"}
        assert _cfg(qbit_info=info).qbit_info == info

    def test_ignore_movies_in_radarr_default_and_override(self) -> None:
        assert _cfg().ignore_movies_in_radarr is False
        assert _cfg(ignore_movies_in_radarr=True).ignore_movies_in_radarr is True

    def test_required_connection_properties_return_value(self) -> None:
        cfg = _cfg(
            sonarr_url="http://s",
            sonarr_api_key="sk",
            radarr_url="http://r",
            radarr_api_key="rk",
        )
        assert cfg.sonarr_url == "http://s"
        assert cfg.sonarr_api_key == "sk"
        assert cfg.radarr_url == "http://r"
        assert cfg.radarr_api_key == "rk"

    def test_required_connection_properties_raise_when_absent(self) -> None:
        cfg = _cfg()
        for getter in ("sonarr_url", "sonarr_api_key", "radarr_url", "radarr_api_key"):
            with pytest.raises(ValueError):
                getattr(cfg, getter)

    def test_optional_radarr_properties_default_none(self) -> None:
        assert _cfg().radarr_url_optional is None
        assert _cfg().radarr_api_key_optional is None
        cfg = _cfg(radarr_url="http://r", radarr_api_key="rk")
        assert cfg.radarr_url_optional == "http://r"
        assert cfg.radarr_api_key_optional == "rk"


class TestImportSettings:
    def test_defaults_when_absent(self) -> None:
        cfg = _cfg()
        assert cfg.import_wait_mode is ImportWaitMode.OFF
        assert cfg.import_wait_timeout == 3600
        assert cfg.import_poll_interval == 30
        assert cfg.import_mode == "auto"
        assert cfg.import_default_quality is None
        assert cfg.import_languages_dual == ["Japanese", "English"]
        assert cfg.import_languages_single == ["Japanese"]
        assert cfg.import_pending_max_age_days == 14

    def test_explicit_values_override(self) -> None:
        cfg = _cfg(
            import_wait_timeout=120,
            import_poll_interval=5,
            import_mode="copy",
            import_default_quality="Bluray-2160p",
            import_languages_dual=["English"],
            import_languages_single=["German"],
            import_pending_max_age_days=30,
        )
        assert cfg.import_wait_timeout == 120
        assert cfg.import_poll_interval == 5
        assert cfg.import_mode == "copy"
        assert cfg.import_default_quality == "Bluray-2160p"
        assert cfg.import_languages_dual == ["English"]
        assert cfg.import_languages_single == ["German"]
        assert cfg.import_pending_max_age_days == 30

    def test_import_wait_mode_coerces_valid_string(self) -> None:
        assert _cfg(import_wait_mode="hybrid").import_wait_mode is ImportWaitMode.HYBRID
        assert _cfg(import_wait_mode="deferred").import_wait_mode is ImportWaitMode.DEFERRED
        assert _cfg(import_wait_mode="blocking").import_wait_mode is ImportWaitMode.BLOCKING
        assert _cfg(import_wait_mode="off").import_wait_mode is ImportWaitMode.OFF

    def test_import_wait_mode_invalid_falls_back_to_off(self) -> None:
        assert _cfg(import_wait_mode="bogus").import_wait_mode is ImportWaitMode.OFF
