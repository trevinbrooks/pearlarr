# pyright: strict, reportPrivateUsage=false
# reportPrivateUsage: the chain-shape and template-grammar pins are white-box
# by design - they guard private wiring no behavior can observe until v2 exists.
"""The config schema-migration chain, the template splice, and the file rewrite.

`tests/test_config.py::TestSchemaMigration` pins the load-path integration
(in-memory migration + reporting). This module pins the pieces: version
detection, each v0 fold, the comment-preserving template splice, and
`upgrade_config_file`'s backup + atomic-rewrite contract.
"""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pearlarr import config as config_mod
from pearlarr import config_migrations
from pearlarr.config import (
    AppConfig,
    ConfigRewriteError,
    ConfigUpgrade,
    starter_template_text,
    upgrade_config_file,
)
from pearlarr.config_migrations import (
    CONFIG_VERSION,
    declared_version,
    migrate_mapping,
    render_migrated_config,
)
from pearlarr.json_narrow import is_json_obj
from pearlarr.seadex_types import Json


class TestDeclaredVersion:
    """Only an absent key means pre-versioning. Anything not a non-negative int is left to validation."""

    def test_version_shapes(self) -> None:
        assert declared_version({}) == 0
        assert declared_version({"config_version": 3}) == 3
        assert declared_version({"config_version": 0}) == 0
        # Blank means default (like every blank key), bools are ints to Python
        # but not versions, strings and negatives are validation's problem.
        assert declared_version({"config_version": None}) is None
        assert declared_version({"config_version": True}) is None
        assert declared_version({"config_version": "1"}) is None
        assert declared_version({"config_version": -3}) is None


class TestMigrateMapping:
    """Each v0 fold fires only on its historical spelling and lands on the old runtime behavior."""

    def test_current_and_newer_mappings_are_untouched(self) -> None:
        for version in (CONFIG_VERSION, CONFIG_VERSION + 1):
            mapping: dict[str, Json] = {"config_version": version, "seadex": {"private_releases": "allow"}}
            assert migrate_mapping(mapping) is None
            assert mapping == {"config_version": version, "seadex": {"private_releases": "allow"}}

    def test_v0_folds_removed_schema_and_nothing_else(self) -> None:
        # Only removed keys/values fold. A never-valid value (the typo'd mode)
        # passes through untouched for validation to reject by name.
        mapping: dict[str, Json] = {
            "seadex": {"public_only": False, "want_best": False},
            "imports": {"mode": "hardlink"},
        }
        outcome = migrate_mapping(mapping)
        assert outcome is not None
        assert mapping == {
            "config_version": CONFIG_VERSION,
            "seadex": {"want_best": False},
            "imports": {"mode": "hardlink"},
        }
        assert len(outcome.notes) == 1

    def test_v0_tolerates_absent_and_malformed_groups(self) -> None:
        # A group that is not a mapping is validation's complaint, not a crash here.
        mapping: dict[str, Json] = {"seadex": 5}
        outcome = migrate_mapping(mapping)
        assert outcome is not None
        assert mapping["seadex"] == 5
        assert mapping["config_version"] == CONFIG_VERSION

    def test_the_chain_is_contiguous_and_ends_at_the_current_version(self) -> None:
        # Declaration order IS application order: pin that the steps run 1..N
        # with no gap, and that bumping CONFIG_VERSION without its step (or
        # reordering the tuple) fails here instead of mis-migrating in the field.
        versions = tuple(step.to_version for step in config_migrations._MIGRATIONS)
        assert versions == tuple(range(1, CONFIG_VERSION + 1))


class TestRenderMigratedConfig:
    """The splice keeps the template's docs and defaults. Explicit values take over their lines."""

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
        # A value needing quoting is re-quoted, not corrupted. false overrides
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

    def test_long_and_multiline_scalars_survive_the_splice(self) -> None:
        # yaml.safe_dump folds plain scalars at ~80 columns. A single-line
        # splice must never truncate (a password!) or emit an unparseable
        # fragment. Line breaks force the escaped double-quoted style.
        long_value = "word " * 30 + "end"
        mapping: dict[str, Json] = {
            "qbittorrent": {"password": long_value, "username": "line one\nline two"},
        }
        parsed: object = yaml.safe_load(render_migrated_config(starter_template_text(), mapping))
        assert is_json_obj(parsed)
        qbit = parsed["qbittorrent"]
        assert is_json_obj(qbit)
        assert qbit["password"] == long_value
        assert qbit["username"] == "line one\nline two"

    def test_every_template_line_matches_a_splice_shape(self) -> None:
        # The splice regexes encode the generator's line grammar (top-level
        # keys, two-space fields, comments, blanks). A generator format change
        # (wider indent, block defaults, new line shapes) must fail HERE, not
        # silently stop matching and revert user values to template defaults.
        for line in starter_template_text().splitlines():
            assert (
                not line.strip()
                or line.lstrip().startswith("#")
                or config_migrations._TOP_KEY.match(line)
                or config_migrations._FIELD_KEY.match(line)
            ), f"template line matches no splice shape: {line!r}"


class TestUpgradeConfigFile:
    """Backup first, template rewrite second, and never touch what is current or unreadable."""

    def _write(self, tmp_path: Path, text: str) -> str:
        path = tmp_path / "config.yml"
        # Bytes, not text: the backup assert compares LF bytes exactly, and
        # write_text would land CRLF on Windows.
        path.write_bytes(text.encode())
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
        # The backup is the previous bytes exactly. The rewrite is the annotated
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
        # Both files carry API keys. Neither may land group/other-readable.
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

    def test_a_long_secret_survives_the_rewrite_end_to_end(self, tmp_path: Path) -> None:
        # Regression: yaml's default scalar folding once truncated any spaced
        # value past ~80 columns during the splice - for a password, silent
        # corruption the success message would have papered over.
        password = "correct horse battery staple " * 5 + "end"
        path = self._write(tmp_path, f"seadex:\n  public_only: true\nqbittorrent:\n  password: '{password}'\n")
        upgrade = upgrade_config_file(path)
        assert upgrade.migration is not None
        loaded = AppConfig.load(path)
        assert loaded.qbittorrent.password is not None
        assert loaded.qbittorrent.password.get_secret_value() == password

    def test_a_render_that_does_not_round_trip_is_refused_before_any_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The pre-replace re-parse is the last line of defense against a splice
        # defect: refuse cleanly, write nothing (not even the backup).
        text = "seadex:\n  public_only: true\n"
        path = self._write(tmp_path, text)

        def corrupt_render(template: str, config: object) -> str:
            return "seadex:\n  want_best: false\n"

        monkeypatch.setattr(config_mod, "render_migrated_config", corrupt_render)
        with pytest.raises(ConfigRewriteError, match="left untouched"):
            upgrade_config_file(path)
        assert Path(path).read_text(encoding="utf-8") == text
        assert not os.path.exists(path + ".bak")
        assert not os.path.exists(path + ".tmp")

    def test_a_failed_swap_leaves_no_secret_bearing_temp_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ENOSPC/permission failure mid-swap: the config survives untouched and
        # the torn .tmp (it carries API keys) is removed. The backup - a
        # faithful copy of the still-intact config - may remain.
        text = "seadex:\n  private_releases: allow\n"
        path = self._write(tmp_path, text)

        def refuse_replace(src: str, dst: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(config_mod.os, "replace", refuse_replace)
        with pytest.raises(OSError, match="disk full"):
            upgrade_config_file(path)
        assert Path(path).read_text(encoding="utf-8") == text
        assert not os.path.exists(path + ".tmp")
