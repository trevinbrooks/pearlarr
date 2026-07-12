# pyright: strict
"""The config schema-migration chain, the template splice, and the file rewrite.

`tests/test_config.py::TestSchemaMigration` pins the load-path integration
(in-memory migration + reporting); this module pins the pieces: version
detection, each v0 fold, the comment-preserving template splice, and
`upgrade_config_file`'s backup + atomic-rewrite contract.
"""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pearlarr.modules.config import (
    AppConfig,
    ConfigUpgrade,
    starter_template_text,
    upgrade_config_file,
)
from pearlarr.modules.config_migrations import (
    CONFIG_VERSION,
    declared_version,
    migrate_mapping,
    render_migrated_config,
)
from pearlarr.modules.json_narrow import is_json_obj
from pearlarr.modules.seadex_types import Json


class TestDeclaredVersion:
    """Absent/blank means pre-versioning (0); non-int garbage is not a version at all."""

    def test_version_shapes(self) -> None:
        assert declared_version({}) == 0
        assert declared_version({"config_version": None}) == 0
        assert declared_version({"config_version": 3}) == 3
        # bools are ints to Python but not versions; strings are validation's problem.
        assert declared_version({"config_version": True}) is None
        assert declared_version({"config_version": "1"}) is None


class TestMigrateMapping:
    """Each v0 fold fires only on its historical spelling and lands on the old runtime behavior."""

    def test_current_and_newer_mappings_are_untouched(self) -> None:
        for version in (CONFIG_VERSION, CONFIG_VERSION + 1):
            mapping: dict[str, Json] = {"config_version": version, "seadex": {"private_releases": "allow"}}
            assert migrate_mapping(mapping) is None
            assert mapping == {"config_version": version, "seadex": {"private_releases": "allow"}}

    def test_v0_folds_only_what_it_recognizes(self) -> None:
        mapping: dict[str, Json] = {
            "seadex": {"public_only": False, "want_best": False},
            "advanced": {"log_level": "warning", "sleep_time": 0},
            "imports": {"mode": "move"},
        }
        outcome = migrate_mapping(mapping)
        assert outcome is not None
        # Valid values survive: `warning` is a level (case-insensitively), `move` a mode.
        assert mapping == {
            "config_version": CONFIG_VERSION,
            "seadex": {"want_best": False},
            "advanced": {"log_level": "warning", "sleep_time": 0},
            "imports": {"mode": "move"},
        }
        assert len(outcome.notes) == 1

    def test_v0_tolerates_absent_and_malformed_groups(self) -> None:
        # A group that is not a mapping is validation's complaint, not a crash here.
        mapping: dict[str, Json] = {"seadex": 5}
        outcome = migrate_mapping(mapping)
        assert outcome is not None
        assert mapping["seadex"] == 5
        assert mapping["config_version"] == CONFIG_VERSION


class TestRenderMigratedConfig:
    """The splice keeps the template's docs and defaults; explicit values take over their lines."""

    def test_unset_keys_keep_the_template_lines(self) -> None:
        rendered = render_migrated_config(starter_template_text(), {"config_version": CONFIG_VERSION})
        # Sample defaults and docs comments survive untouched.
        assert "  verify_ssl: true" in rendered
        assert "# API key of the instance" in rendered
        assert f"config_version: {CONFIG_VERSION}" in rendered

    def test_explicit_values_round_trip_through_yaml(self) -> None:
        mapping: dict[str, Json] = {
            "config_version": CONFIG_VERSION,
            "sonarr": {"url": "http://s:8989", "api_key": "key: with colon", "verify_ssl": False},
            "qbittorrent": {"tags": ["anime", "seadex"], "options": {"REQUESTS_ARGS": {"timeout": 5}}},
            "imports": {"languages_dual": [], "default_quality": None},
        }
        parsed: object = yaml.safe_load(render_migrated_config(starter_template_text(), mapping))
        assert is_json_obj(parsed)
        assert parsed["config_version"] == CONFIG_VERSION
        sonarr = parsed["sonarr"]
        assert is_json_obj(sonarr)
        # A value needing quoting is re-quoted, not corrupted; false overrides
        # the template's sample `true`.
        assert sonarr["api_key"] == "key: with colon"
        assert sonarr["verify_ssl"] is False
        qbit = parsed["qbittorrent"]
        assert is_json_obj(qbit)
        assert qbit["tags"] == ["anime", "seadex"]
        assert qbit["options"] == {"REQUESTS_ARGS": {"timeout": 5}}
        imports = parsed["imports"]
        assert is_json_obj(imports)
        # An explicit empty list and an explicit blank both survive as themselves.
        assert imports["languages_dual"] == []
        assert imports["default_quality"] is None


class TestUpgradeConfigFile:
    """Backup first, template rewrite second, and never touch what is current or unreadable."""

    def _write(self, tmp_path: Path, text: str) -> str:
        path = tmp_path / "config.yml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_old_file_is_rewritten_with_backup_and_equal_meaning(self, tmp_path: Path) -> None:
        text = "seadex:\n  public_only: true\nsonarr:\n  url: http://s\n  api_key: hunter2\n"
        path = self._write(tmp_path, text)
        before = AppConfig.load(path)

        upgrade = upgrade_config_file(path)

        assert upgrade.migration is not None
        assert upgrade.migration.from_version == 0
        backup = path + ".bak"
        assert upgrade.backup_path == backup
        # The backup is the previous bytes exactly; the rewrite is the annotated
        # template carrying the file's values, so the effective config is unchanged.
        assert Path(backup).read_bytes() == text.encode()
        rewritten = Path(path).read_text(encoding="utf-8")
        assert "api_key: hunter2" in rewritten
        assert "public_only" not in rewritten
        assert "# API key of the instance" in rewritten
        after = AppConfig.load(path)
        assert after.migration() is None
        assert after.model_dump() == before.model_dump()

    @pytest.mark.skipif(os.name != "posix", reason="mode bits are POSIX-only")
    def test_rewrite_and_backup_are_owner_only(self, tmp_path: Path) -> None:
        # Both files carry API keys; neither may land group/other-readable.
        path = self._write(tmp_path, "seadex:\n  private_releases: allow\n")
        upgrade = upgrade_config_file(path)
        assert upgrade.backup_path is not None
        assert (Path(path).stat().st_mode & 0o777) == 0o600
        assert (Path(upgrade.backup_path).stat().st_mode & 0o777) == 0o600

    def test_current_file_is_left_byte_identical(self, tmp_path: Path) -> None:
        text = f"config_version: {CONFIG_VERSION}\nsonarr:\n  url: http://s\n"
        path = self._write(tmp_path, text)
        assert upgrade_config_file(path) == ConfigUpgrade(migration=None, backup_path=None)
        assert Path(path).read_text(encoding="utf-8") == text
        assert not os.path.exists(path + ".bak")

    def test_an_invalid_file_is_never_rewritten(self, tmp_path: Path) -> None:
        # Refusing to rewrite what this version cannot fully read: no partial
        # writes, no backup, the original untouched.
        for text in ("seadex:\n  public_only: true\ntypo_key: 1\n", "just a scalar\n"):
            path = self._write(tmp_path, text)
            with pytest.raises(ValidationError):
                upgrade_config_file(path)
            assert Path(path).read_text(encoding="utf-8") == text
            assert not os.path.exists(path + ".bak")
