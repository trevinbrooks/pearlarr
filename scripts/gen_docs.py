"""Generate the documentation artifacts whose source of truth is code.

One authored home per fact: config facts live as attribute docstrings on the
pydantic models in ``pearlarr/modules/config.py`` (plus enum member docstrings
and the env-var registry), and this script renders every other surface from
them:

- ``pearlarr/modules/config_sample.yml`` - the starter config template
- ``schemas/config.schema.json`` - JSON Schema for editor validation
- ``docs/configuration.md`` - the generated islands between ``gen:`` markers

Write mode (default) rewrites the artifacts in place; ``--check`` exits
non-zero when any artifact is not byte-identical to what would be generated
(the doc test suite and pre-commit run this). Missing field or enum-member
docstrings are a hard error in both modes.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import inspect
import re
import sys
import textwrap
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version
from json import dumps
from pathlib import Path
from types import UnionType
from typing import Any, Literal, cast, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from pearlarr.modules.config import (
    OTHER_TRACKER_NAMES,
    PRIVATE_TRACKER_NAMES,
    PUBLIC_TRACKER_NAMES,
    AppConfig,
)
from pearlarr.modules.env_registry import ENV_VARS
from pearlarr.modules.paths import PROJECT_URL

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = "pearlarr/modules/config.py"
REGEN_COMMAND = "uv run python scripts/gen_docs.py"

SAMPLE_PATH = REPO_ROOT / "pearlarr" / "modules" / "config_sample.yml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "config.schema.json"
CONFIGURATION_PATH = REPO_ROOT / "docs" / "configuration.md"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "architecture.md"

# Comments in the sample wrap so the whole line stays inside this width.
SAMPLE_WIDTH = 100

# The schema URL is tag-pinned (G8): a user's copied config validates against
# the schema of the version they installed, not whatever main looks like today.
RAW_BASE = PROJECT_URL.replace("https://github.com/", "https://raw.githubusercontent.com/")
SCHEMA_URL = f"{RAW_BASE}/v{version('pearlarr')}/schemas/config.schema.json"

KNOWN_TRACKER_DISPLAY = PUBLIC_TRACKER_NAMES + PRIVATE_TRACKER_NAMES + OTHER_TRACKER_NAMES

# Fields whose effective default is computed at load time (a validator), not the
# static field default: the sample must ship them blank - writing the static
# default out would pin the derived behavior off - and the table says "derived"
# because the docstring already explains the derivation.
DERIVED_FIELDS = frozenset({"notifications.wait_notify"})


class GenerationError(Exception):
    """A documentation source is incomplete or a stitch target is malformed."""


@dataclass(frozen=True)
class LeafDoc:
    """One config field, fully resolved for rendering."""

    key: str
    description: str
    sample_value: str
    table_default: str
    values: tuple[tuple[str, str], ...]
    known: tuple[str, ...]
    default_note: str


@dataclass(frozen=True)
class GroupDoc:
    """One settings group (a nested submodel) and its fields."""

    key: str
    class_name: str
    description: str
    fields: tuple[LeafDoc, ...]


def flatten(text: str) -> str:
    """One-line form of a docstring: all whitespace runs collapse to single spaces."""

    return " ".join(text.split())


def strip_ticks(text: str) -> str:
    """Drop backticks for plain-text surfaces (YAML comments are not markdown)."""

    return text.replace("`", "")


def normalize_ticks(text: str) -> str:
    """Collapse reST-style double backticks to markdown single backticks."""

    return re.sub(r"``([^`]+)``", r"`\1`", text)


def enum_member_docs(enum_cls: type[Enum]) -> dict[str, str]:
    """Attribute docstrings of an enum's members, read from source (lost at runtime)."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(enum_cls)))
    class_def = tree.body[0]
    if not isinstance(class_def, ast.ClassDef):
        raise GenerationError(f"could not parse {enum_cls.__name__} as a class definition")
    docs: dict[str, str] = {}
    member: str | None = None
    for node in class_def.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            member = node.targets[0].id
        elif (
            member is not None
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docs[member] = flatten(inspect.cleandoc(node.value.value))
            member = None
        else:
            member = None
    return docs


def _unwrap_annotation(annotation: object) -> object:
    """Resolve PEP 695 aliases and strip ``| None`` when one real arm remains."""

    ann = getattr(annotation, "__value__", annotation)
    if get_origin(ann) is UnionType:
        args = [arg for arg in get_args(ann) if arg is not type(None)]
        if len(args) == 1:
            ann = getattr(args[0], "__value__", args[0])
    return ann


def field_values(annotation: object) -> tuple[tuple[str, str], ...]:
    """The enumerable values of a field, paired with member docs where they exist.

    Enums contribute their member docstrings; a pure string ``Literal``
    contributes bare values. Anything else (unions of shapes, free-form types)
    enumerates nothing.
    """

    ann = _unwrap_annotation(annotation)
    if isinstance(ann, type) and issubclass(ann, Enum):
        docs = enum_member_docs(ann)
        missing = [member.name for member in ann if not docs.get(member.name)]
        if missing:
            raise GenerationError(f"{ann.__name__} members missing docstrings: {', '.join(missing)}")
        return tuple((str(member.value), docs[member.name]) for member in ann)
    if get_origin(ann) is Literal:
        args = get_args(ann)
        if all(isinstance(arg, str) for arg in args):
            return tuple((cast("str", arg), "") for arg in args)
    return ()


def yaml_scalar(value: object) -> str:
    """Render a scalar default exactly as it belongs in the sample YAML."""

    if isinstance(value, Enum):
        value = value.value
    # A bare-scalar document gets a `...` end marker on its own line; the
    # scalar itself is the first line.
    return yaml.safe_dump(value, default_flow_style=True).partition("\n")[0]


def yaml_flow(value: list[str]) -> str:
    """Render a list default in flow style (``[Japanese, English]``)."""

    return yaml.safe_dump(value, default_flow_style=True).strip()


def build_leaf(group_key: str, key: str, field: FieldInfo) -> LeafDoc:
    """Resolve one field's docs, default rendering, and value enumeration."""

    if not field.description:
        raise GenerationError(f"config field {group_key}.{key} has no attribute docstring")
    description = flatten(field.description)
    default = field.get_default(call_default_factory=True)

    values = field_values(field.annotation)
    known: tuple[str, ...] = ()
    default_note = ""
    sample_value = ""
    table_default = "*(blank)*"

    if f"{group_key}.{key}" in DERIVED_FIELDS:
        table_default = "*(derived)*"
    elif f"{group_key}.{key}" == "seadex.trackers":
        # The one field whose allowed values live outside the type system: the
        # display-cased tracker tuples are the source (KNOWN_TRACKERS is their
        # casefolded shadow).
        known = KNOWN_TRACKER_DISPLAY
        table_default = "all supported trackers"
    elif isinstance(default, (list, tuple)):
        listed = [str(item) for item in cast("list[object] | tuple[object, ...]", default)]
        if listed:
            default_note = yaml_flow(listed)
            table_default = f"`{default_note}`"
    elif isinstance(default, set):
        # Sets iterate in str-hash order, randomized per process: sort for
        # byte-identical output.
        listed = sorted(str(item) for item in cast("set[object]", default))
        if listed:
            default_note = yaml_flow(listed)
            table_default = f"`{default_note}`"
    elif isinstance(default, dict):
        pass  # the only dict defaults are empty escape hatches; stay blank
    elif default is not None and default is not PydanticUndefined:
        sample_value = yaml_scalar(default)
        table_default = f"`{sample_value}`"

    return LeafDoc(
        key=key,
        description=description,
        sample_value=sample_value,
        table_default=table_default,
        values=values,
        known=known,
        default_note=default_note,
    )


def build_groups() -> tuple[GroupDoc, ...]:
    """Walk ``AppConfig`` into renderable group docs, failing on any gap."""

    groups: list[GroupDoc] = []
    for group_key, group_field in AppConfig.model_fields.items():
        if not group_field.description:
            raise GenerationError(f"config group {group_key} has no attribute docstring")
        submodel = group_field.annotation
        if not (isinstance(submodel, type) and issubclass(submodel, BaseModel)):
            raise GenerationError(f"config group {group_key} is not a nested model")
        fields = tuple(build_leaf(group_key, key, field) for key, field in submodel.model_fields.items())
        groups.append(
            GroupDoc(
                key=group_key,
                class_name=submodel.__name__,
                description=flatten(group_field.description),
                fields=fields,
            ),
        )
    return tuple(groups)


def comment_lines(text: str, indent: str) -> list[str]:
    """Wrap plain text into ``# `` comment lines within the sample width."""

    width = SAMPLE_WIDTH - len(indent) - 2
    return [f"{indent}# {line}" for line in textwrap.wrap(text, width=width, break_on_hyphens=False)]


def values_lines(leaf: LeafDoc, indent: str) -> list[str]:
    """The generator-injected value enumeration for a field, if any."""

    lines: list[str] = []
    if leaf.values:
        if all(not doc for _, doc in leaf.values):
            lines.extend(comment_lines(f"Values: {' / '.join(value for value, _ in leaf.values)}", indent))
        else:
            lines.append(f"{indent}# Values:")
            pad = max(len(value) for value, _ in leaf.values)
            for value, doc in leaf.values:
                prefix = f"{indent}#   {value:<{pad}} - "
                wrapped = textwrap.wrap(
                    strip_ticks(doc),
                    width=max(SAMPLE_WIDTH - len(prefix), 40),
                    break_on_hyphens=False,
                )
                lines.append(prefix + wrapped[0])
                lines.extend(f"{indent}#   {' ' * pad}   {cont}" for cont in wrapped[1:])
    if leaf.known:
        lines.extend(comment_lines(f"Known: {', '.join(leaf.known)}", indent))
    if leaf.default_note:
        lines.extend(comment_lines(f"Default: {leaf.default_note}", indent))
    return lines


def render_sample(groups: tuple[GroupDoc, ...]) -> str:
    """The starter config template, every fact drawn from the model tree."""

    lines: list[str] = [
        f"# GENERATED by scripts/gen_docs.py from {CONFIG_SOURCE} - do not edit here; regenerate: {REGEN_COMMAND}",
        f"# yaml-language-server: $schema={SCHEMA_URL}",
    ]
    for group in groups:
        lines.append("")
        lines.extend(comment_lines(strip_ticks(group.description), ""))
        lines.append(f"{group.key}:")
        for index, leaf in enumerate(group.fields):
            if index:
                lines.append("")
            lines.extend(comment_lines(strip_ticks(leaf.description), "  "))
            lines.extend(values_lines(leaf, "  "))
            suffix = f" {leaf.sample_value}" if leaf.sample_value else ""
            lines.append(f"  {leaf.key}:{suffix}")
    return "\n".join(lines) + "\n"


def _normalize_descriptions(node: object) -> None:
    """Flatten and de-reST every ``description`` in a JSON-schema tree, in place."""

    if isinstance(node, dict):
        typed = cast("dict[str, object]", node)
        for key, value in typed.items():
            if key == "description" and isinstance(value, str):
                typed[key] = normalize_ticks(flatten(value))
            else:
                _normalize_descriptions(value)
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            _normalize_descriptions(item)


def render_schema(groups: tuple[GroupDoc, ...]) -> str:
    """The JSON Schema for ``config.yml``, published at a tag-stable path."""

    generated = AppConfig.model_json_schema()
    defs = cast("dict[str, dict[str, Any]]", generated.get("$defs", {}))
    for group in groups:
        # The submodel class docstrings are contributor-facing contracts; the
        # group attribute docstrings are what a config editor should surface.
        defs[group.class_name]["description"] = group.description
    for def_schema in defs.values():
        # Enum class docstrings: only the summary paragraph is for users; the
        # rationale paragraphs are contributor-facing.
        description = def_schema.get("description")
        if "enum" in def_schema and isinstance(description, str):
            def_schema["description"] = description.split("\n\n")[0]
    generated.pop("title", None)
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_URL,
        "$comment": f"GENERATED by scripts/gen_docs.py from {CONFIG_SOURCE}; do not edit; regenerate: {REGEN_COMMAND}",
        "title": "Pearlarr configuration",
        **generated,
    }
    _normalize_descriptions(schema)
    return dumps(schema, indent=2, ensure_ascii=False) + "\n"


def md_cell(text: str) -> str:
    """Escape a fragment for use inside a markdown table cell."""

    return text.replace("|", "\\|")


def render_group_table(group: GroupDoc) -> str:
    """The field table for one settings group."""

    rows = ["| Key | Default | Values | Description |", "| --- | --- | --- | --- |"]
    for leaf in group.fields:
        values = " / ".join(f"`{value}`" for value, _ in leaf.values)
        if leaf.known:
            values = ", ".join(f"`{name}`" for name in leaf.known)
        description = leaf.description
        member_docs = " ".join(f"`{value}`: {doc}" for value, doc in leaf.values if doc)
        if member_docs:
            description = f"{description} {member_docs}"
        rows.append(f"| `{leaf.key}` | {md_cell(leaf.table_default)} | {md_cell(values)} | {md_cell(description)} |")
    return "\n".join(rows) + "\n"


def render_env_table() -> str:
    """The environment-variable table, from the registry."""

    scope_display = {"app": "application", "docker": "Docker entrypoint"}
    rows = ["| Variable | Read by | Meaning |", "| --- | --- | --- |"]
    rows.extend(f"| `{var.name}` | {scope_display[var.scope]} | {md_cell(var.description)} |" for var in ENV_VARS)
    return "\n".join(rows) + "\n"


def stitch(document: str, island: str, content: str, source: str) -> str:
    """Replace one generated island of a stitched markdown document."""

    open_pattern = re.compile(rf"^<!-- gen:{re.escape(island)}\b[^\n]*-->$", re.MULTILINE)
    close_marker = f"<!-- /gen:{island} -->"
    open_match = open_pattern.search(document)
    close_index = document.find(close_marker)
    if open_match is None or close_index < 0 or close_index < open_match.end():
        raise GenerationError(f"island {island} is missing or malformed (check its gen: markers)")
    opener = (
        f"<!-- gen:{island} - GENERATED by scripts/gen_docs.py from {source}; "
        f"do not edit between the markers; regenerate: {REGEN_COMMAND} -->"
    )
    return document[: open_match.start()] + opener + "\n" + content + document[close_index:]


def render_configuration(groups: tuple[GroupDoc, ...]) -> str:
    """docs/configuration.md with every generated island refreshed."""

    if not CONFIGURATION_PATH.exists():
        raise GenerationError(f"{CONFIGURATION_PATH} does not exist; author its prose skeleton first")
    document = CONFIGURATION_PATH.read_text(encoding="utf-8")
    document = stitch(document, "env-vars", render_env_table(), "pearlarr/modules/env_registry.py")
    for group in groups:
        document = stitch(document, f"group-{group.key}", render_group_table(group), CONFIG_SOURCE)
    return document


def render_contributing() -> str:
    """CONTRIBUTING.md with its env-var island refreshed."""

    if not CONTRIBUTING_PATH.exists():
        raise GenerationError(f"{CONTRIBUTING_PATH} does not exist")
    document = CONTRIBUTING_PATH.read_text(encoding="utf-8")
    return stitch(document, "env-vars", render_env_table(), "pearlarr/modules/env_registry.py")


def collect_invariants() -> tuple[tuple[str, str], ...]:
    """Every ``# Invariant:`` comment block in the package, as (module, text) pairs.

    A block runs from its ``# Invariant:`` line through the directly following
    comment lines; blocks appear in path order, then file order.
    """

    found: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "pearlarr").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("# Invariant:"):
                block = [stripped.removeprefix("# Invariant:").strip()]
                index += 1
                while index < len(lines):
                    cont = lines[index].strip()
                    if cont.startswith("# Invariant:") or not cont.startswith("#"):
                        break
                    block.append(cont.removeprefix("#").strip())
                    index += 1
                # as_posix: the generated index must not grow \ separators on Windows.
                found.append((path.relative_to(REPO_ROOT).as_posix(), flatten(" ".join(block))))
            else:
                index += 1
    return tuple(found)


def render_invariant_index() -> str:
    """The invariant index for architecture.md, from the enforcement-site comments."""

    invariants = collect_invariants()
    if not invariants:
        raise GenerationError("no # Invariant: comments found under pearlarr/")
    return "\n".join(f"- `{module}` - {md_cell(text)}" for module, text in invariants) + "\n"


def render_architecture() -> str:
    """docs/architecture.md with its invariant island refreshed."""

    if not ARCHITECTURE_PATH.exists():
        raise GenerationError(f"{ARCHITECTURE_PATH} does not exist; author its prose skeleton first")
    document = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    return stitch(document, "invariants", render_invariant_index(), "the # Invariant: comments in pearlarr/")


def artifacts() -> dict[Path, str]:
    """Every generated artifact, rendered."""

    groups = build_groups()
    return {
        SAMPLE_PATH: render_sample(groups),
        SCHEMA_PATH: render_schema(groups),
        CONFIGURATION_PATH: render_configuration(groups),
        CONTRIBUTING_PATH: render_contributing(),
        ARCHITECTURE_PATH: render_architecture(),
    }


def main() -> int:
    """Entry point: write the artifacts, or verify them with ``--check``."""

    parser = argparse.ArgumentParser(description="Generate documentation artifacts from code.")
    parser.add_argument("--check", action="store_true", help="verify artifacts are current; exit 1 on drift")
    args = parser.parse_args()

    try:
        rendered = artifacts()
    except GenerationError as error:
        sys.stderr.write(f"gen_docs: {error}\n")
        return 2

    drifted: list[Path] = []
    for path, content in rendered.items():
        on_disk = path.read_text(encoding="utf-8") if path.exists() else ""
        if content == on_disk:
            continue
        if args.check:
            drifted.append(path)
            diff = difflib.unified_diff(
                on_disk.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=str(path.relative_to(REPO_ROOT)),
                tofile="generated",
            )
            sys.stderr.writelines(list(diff)[:40])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            sys.stdout.write(f"gen_docs: wrote {path.relative_to(REPO_ROOT)}\n")

    if drifted:
        names = ", ".join(str(path.relative_to(REPO_ROOT)) for path in drifted)
        sys.stderr.write(f"gen_docs: stale generated docs: {names} - regenerate: {REGEN_COMMAND}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
