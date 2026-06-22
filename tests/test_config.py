"""Characterization tests for ``AppConfig``.

Pins the config-file lifecycle and the typed-settings normalization moved out of
``SeaDexArr.__init__`` / ``verify_config`` in Phase 2 (see ``REFACTOR_PLAN.md``).
"""

import hashlib

import pytest
import yaml

from seadexarr.modules.config import PRIVATE_TRACKERS, PUBLIC_TRACKERS, AppConfig


def _cfg(**data: object) -> AppConfig:
    """An ``AppConfig`` over an in-memory data dict (no file load)."""

    return AppConfig(path="unused.yml", arr="sonarr", data=dict(data))


class TestFileLifecycle:
    def test_missing_config_copies_template_and_raises(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path), "sonarr")
        # The bundled template was copied into place for the user to edit.
        assert cfg_path.exists()
        assert "sonarr_url" in yaml.safe_load(cfg_path.read_text())

    def test_load_preserves_user_values(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("sonarr_url: http://x\npublic_only: false\n")
        cfg = AppConfig.load(str(cfg_path), "sonarr")
        assert cfg.data["sonarr_url"] == "http://x"
        assert cfg.public_only is False

    def test_checksum_matches_file_bytes(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_bytes(b"public_only: true\n")
        cfg = AppConfig(path=str(cfg_path), arr="sonarr", data={})
        assert cfg.checksum() == hashlib.md5(b"public_only: true\n").hexdigest()


class TestTemplateSync:
    def test_partial_config_is_rewritten_in_template_order(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        # Keys present but out of template order -> triggers the rewrite.
        cfg_path.write_text("public_only: false\nsonarr_url: http://x\n")
        cfg = AppConfig.load(str(cfg_path), "sonarr")
        # First template key leads, a previously-absent key is now present, and
        # the user's values survive the merge.
        assert next(iter(cfg.data)) == "sonarr_url"
        assert "log_level" in cfg.data
        assert cfg.public_only is False
        assert cfg.data["sonarr_url"] == "http://x"

    def test_template_order_config_not_rewritten(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yml"
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(cfg_path), "sonarr")  # copies the template
        before = cfg_path.read_bytes()
        AppConfig.load(str(cfg_path), "sonarr")  # already in template order
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
        sonarr = AppConfig(path="x", arr="sonarr", data=data)
        radarr = AppConfig(path="x", arr="radarr", data=data)
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
