# pyright: strict
"""Config documentation pipeline: generated artifacts current, docstrings complete and clean.

The generator itself hard-fails on a missing field or enum-member docstring;
these tests add the byte-equality drift gate and the content rules the
generator cannot judge (no restated defaults, env registry parity).
"""

import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from pearlarr.modules.config import AppConfig
from pearlarr.modules.env_registry import ENV_VARS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _leaf_fields() -> list[tuple[str, FieldInfo]]:
    """Every (dotted key, field) pair of the config tree, groups included."""

    pairs: list[tuple[str, FieldInfo]] = []
    for group_key, group_field in AppConfig.model_fields.items():
        pairs.append((group_key, group_field))
        annotation = group_field.annotation
        assert isinstance(annotation, type) and issubclass(annotation, BaseModel)
        pairs.extend((f"{group_key}.{key}", field) for key, field in annotation.model_fields.items())
    return pairs


def test_generated_docs_are_current() -> None:
    # The single-source pipeline's drift gate: config_sample.yml, the JSON
    # schema, and configuration.md's islands must match what the models render.
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_docs.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"generated docs stale or invalid:\n{result.stdout}{result.stderr}"


def test_every_config_field_has_a_docstring() -> None:
    undocumented = [dotted for dotted, field in _leaf_fields() if not (field.description or "").strip()]
    assert undocumented == []


def test_config_docstrings_never_restate_defaults() -> None:
    # The generator injects defaults and allowed values; prose restating them
    # is the drift the single-source pipeline exists to kill.
    banned = re.compile(r"[Dd]efaults? to|[Dd]efault:|\(default")
    offenders = [dotted for dotted, field in _leaf_fields() if field.description and banned.search(field.description)]
    assert offenders == []


def test_env_registry_matches_the_tree() -> None:
    # Every PEARLARR_* variable mentioned anywhere (code, Docker, docs) is
    # registered, and every registered variable is actually mentioned.
    # The one exemption: the hypothetical the naming-scheme rule is explained with.
    hypothetical = {"PEARLARR_SONARR__URL"}
    scanned: set[str] = set()
    files = [
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "docker-compose.example.yml",
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        *sorted((REPO_ROOT / "docker").glob("*")),
        *sorted((REPO_ROOT / "pearlarr").rglob("*.py")),
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ]
    for path in files:
        if path.is_file():
            scanned.update(re.findall(r"PEARLARR_[A-Z_]+", path.read_text(encoding="utf-8")))
    assert scanned - hypothetical == {var.name for var in ENV_VARS}
