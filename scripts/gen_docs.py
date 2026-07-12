"""Generate the documentation artifacts whose source of truth is code.

One authored home per fact: config facts live as attribute docstrings on the
pydantic models in `pearlarr/modules/config.py` (plus enum member docstrings
and the env-var registry), and this script renders every other surface from
them:

- `pearlarr/modules/config_sample.yml` - the starter config template
- `schemas/config.schema.json` - JSON Schema for editor validation
- `docs/configuration.md` - the generated islands between `gen:` markers
- `docs/cli.md` - the command reference, from the typer app
- `docs/output.md` - the JSON event catalog island, from the event
  vocabulary run through the real JSON serializer

Write mode (default) rewrites the artifacts in place; `--check` exits
non-zero when any artifact is not byte-identical to what would be generated
(the doc test suite and pre-commit run this). Missing field or enum-member
docstrings are a hard error in both modes.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import inspect
import io
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import version
from json import dumps, loads
from pathlib import Path
from types import UnionType
from typing import Any, Literal, cast, get_args, get_origin

import typer.core
import typer.main
import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from pearlarr.modules.cli import pearlarr_cli
from pearlarr.modules.config import (
    OTHER_TRACKER_NAMES,
    PRIVATE_TRACKER_NAMES,
    PUBLIC_TRACKER_NAMES,
    AppConfig,
    Arr,
)
from pearlarr.modules.env_registry import ENV_VARS
from pearlarr.modules.json_narrow import is_json_obj
from pearlarr.modules.log import EntryState
from pearlarr.modules.manual_import import Outcome, OutcomeCategory
from pearlarr.modules.output import JsonRenderer
from pearlarr.modules.output import events as ev
from pearlarr.modules.paths import PROJECT_URL

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = "pearlarr/modules/config.py"
REGEN_COMMAND = "uv run python scripts/gen_docs.py"

SAMPLE_PATH = REPO_ROOT / "pearlarr" / "modules" / "config_sample.yml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "config.schema.json"
CONFIGURATION_PATH = REPO_ROOT / "docs" / "configuration.md"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "architecture.md"
CLI_PATH = REPO_ROOT / "docs" / "cli.md"
OUTPUT_PATH = REPO_ROOT / "docs" / "output.md"
CLI_SOURCE = "pearlarr/modules/cli.py"
EVENTS_SOURCE = "pearlarr/modules/output/events.py + textline.py"

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


def reject_double_ticks(text: str) -> str:
    """Refuse reST-style double backticks - the compiled dialect is single-backtick only."""

    if "``" in text:
        raise GenerationError(f"double backticks in a config docstring: {text[:60]!r}")
    return text


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
    """Resolve PEP 695 aliases and strip `| None` when one real arm remains."""

    ann = getattr(annotation, "__value__", annotation)
    if get_origin(ann) is UnionType:
        args = [arg for arg in get_args(ann) if arg is not type(None)]
        if len(args) == 1:
            ann = getattr(args[0], "__value__", args[0])
    return ann


def field_values(annotation: object) -> tuple[tuple[str, str], ...]:
    """The enumerable values of a field, paired with member docs where they exist.

    Enums contribute their member docstrings; a pure string `Literal`
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
    """Render a list default in flow style (`[Japanese, English]`)."""

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
    """Walk `AppConfig` into renderable group docs, failing on any gap."""

    groups: list[GroupDoc] = []
    for group_key, group_field in AppConfig.model_fields.items():
        if group_key == "config_version":
            continue  # the one top-level scalar; render_sample places it by hand
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
    """Wrap plain text into `# ` comment lines within the sample width."""

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
    version = build_leaf("", "config_version", AppConfig.model_fields["config_version"])
    lines.append("")
    lines.extend(comment_lines(strip_ticks(version.description), ""))
    lines.append(f"config_version: {version.sample_value}")
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
    """Flatten and de-reST every `description` in a JSON-schema tree, in place."""

    if isinstance(node, dict):
        typed = cast("dict[str, object]", node)
        for key, value in typed.items():
            if key == "description" and isinstance(value, str):
                typed[key] = reject_double_ticks(flatten(value))
            else:
                _normalize_descriptions(value)
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            _normalize_descriptions(item)


def render_schema(groups: tuple[GroupDoc, ...]) -> str:
    """The JSON Schema for `config.yml`, published at a tag-stable path."""

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
    # The root description gets the same summary-only treatment: the rationale
    # paragraphs of the AppConfig docstring are contributor-facing.
    root_description = generated.get("description")
    if isinstance(root_description, str):
        generated["description"] = root_description.split("\n\n")[0]
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
    """Every `# Invariant:` comment block in the package, as (module, text) pairs.

    A block runs from its `# Invariant:` line through the directly following
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


# --- docs/cli.md: the command reference, from the typer app -------------------------


@dataclass(frozen=True)
class OptionDoc:
    """One command option, resolved for the reference table."""

    flags: str
    value: str
    description: str


@dataclass(frozen=True)
class CommandDoc:
    """One command (or command group) as a reference section."""

    path: str
    depth: int
    help: str
    options: tuple[OptionDoc, ...]
    subcommands: tuple[tuple[str, str], ...]


# Exit codes are wire facts of the click/typer stack plus our own exits
# (verified empirically); no runtime surface enumerates them, so they are
# authored here, next to the reference they render into.
EXIT_CODE_ROWS = (
    ("0", "Success. A scheduled loop stopped with SIGTERM also exits 0 (a clean stop)."),
    ("1", "Failure: invalid or missing configuration, a refused selection, a failed run, or a failed command."),
    ("2", "Usage error: an unknown command, option, or option value."),
    ("130", "Interrupted with Ctrl-C."),
)


def _help_paragraphs(raw: str) -> str:
    """A command's help text as markdown paragraphs, each reflowed to one line."""

    return "\n\n".join(flatten(paragraph) for paragraph in raw.split("\n\n") if paragraph.strip())


def _first_paragraph(raw: str) -> str:
    """The one-line form of a help text's first paragraph (group listings)."""

    return _help_paragraphs(raw).split("\n\n")[0]


# typer builds every node as one of these two (its vendored click is private,
# so the walker narrows to typer's public classes instead).
type TyperNode = typer.core.TyperCommand | typer.core.TyperGroup


def _typer_node(path: str, command: object) -> TyperNode:
    """Narrow a tree member to typer's public command classes, or fail loudly."""

    if not isinstance(command, (typer.core.TyperCommand, typer.core.TyperGroup)):
        raise GenerationError(f"command {path} is not a typer-built command")
    return command


def _option_value(param: typer.core.TyperOption) -> str:
    """The Value cell: enumerated choices, the metavar, or nothing for a pure flag."""

    choices: object = getattr(param.type, "choices", None)
    if isinstance(choices, tuple):
        return " / ".join(f"`{choice}`" for choice in cast("tuple[str, ...]", choices))
    if param.is_flag:
        return ""
    return f"`{param.metavar or param.type.name.upper()}`"


def _option_docs(path: str, command: TyperNode) -> tuple[OptionDoc, ...]:
    """The visible options of one command, `--help` itself excluded."""

    options: list[OptionDoc] = []
    for param in command.params:
        if not isinstance(param, typer.core.TyperOption) or param.hidden or param.name == "help":
            continue
        if not param.help:
            raise GenerationError(f"option {param.opts[0]} of {path} has no help text")
        flags = ", ".join(f"`{opt}`" for opt in param.opts)
        if param.secondary_opts:
            flags += " / " + ", ".join(f"`{opt}`" for opt in param.secondary_opts)
        options.append(OptionDoc(flags=flags, value=_option_value(param), description=flatten(param.help)))
    return tuple(options)


def _command_doc(path: str, depth: int, command: TyperNode) -> CommandDoc:
    """One command resolved into its reference section."""

    if not command.help:
        raise GenerationError(f"command {path} has no help text")
    subcommands: tuple[tuple[str, str], ...] = ()
    if isinstance(command, typer.core.TyperGroup):
        for name, sub in command.commands.items():
            help_text = _typer_node(f"{path} {name}", sub).help
            if not help_text:
                raise GenerationError(f"command {path} {name} has no help text")
            subcommands = (*subcommands, (f"{path} {name}", _first_paragraph(help_text)))
    return CommandDoc(
        path=path,
        depth=depth,
        help=_help_paragraphs(command.help),
        options=_option_docs(path, command),
        subcommands=subcommands,
    )


def build_command_docs() -> tuple[CommandDoc, ...]:
    """Walk the built command tree into reference sections, groups before their leaves."""

    root = typer.main.get_command(pearlarr_cli)
    if not isinstance(root, typer.core.TyperGroup):
        raise GenerationError("the pearlarr CLI did not build as a command group")
    docs: list[CommandDoc] = [_command_doc("pearlarr", 2, root)]
    for name, sub in root.commands.items():
        node = _typer_node(f"pearlarr {name}", sub)
        docs.append(_command_doc(f"pearlarr {name}", 2, node))
        if isinstance(node, typer.core.TyperGroup):
            docs.extend(
                _command_doc(f"pearlarr {name} {leaf_name}", 3, _typer_node(f"pearlarr {name} {leaf_name}", leaf))
                for leaf_name, leaf in node.commands.items()
            )
    return tuple(docs)


def _options_table(options: tuple[OptionDoc, ...]) -> list[str]:
    if not options:
        return []
    rows = ["| Option | Value | Description |", "| --- | --- | --- |"]
    rows.extend(f"| {opt.flags} | {md_cell(opt.value)} | {md_cell(opt.description)} |" for opt in options)
    return [*rows, ""]


def _subcommands_table(subcommands: tuple[tuple[str, str], ...]) -> list[str]:
    if not subcommands:
        return []
    rows = ["| Command | Description |", "| --- | --- |"]
    rows.extend(f"| `{path}` | {md_cell(summary)} |" for path, summary in subcommands)
    return [*rows, ""]


def render_cli() -> str:
    """docs/cli.md: every command and option, plus the exit-code contract."""

    lines = [
        f"<!-- GENERATED by scripts/gen_docs.py from {CLI_SOURCE} - do not edit; regenerate: {REGEN_COMMAND} -->",
        "",
        "# Command-line reference",
        "",
        "Every Pearlarr command, generated from the CLI definition. `pearlarr <command> --help`",
        "shows the same text for the version you actually have; every command accepts `-h`/`--help`.",
        "",
        "Exit codes, for scripts and schedulers:",
        "",
        "| Code | Meaning |",
        "| --- | --- |",
        *(f"| `{code}` | {meaning} |" for code, meaning in EXIT_CODE_ROWS),
        "",
    ]
    for doc in build_command_docs():
        lines.append(f"{'#' * doc.depth} `{doc.path}`")
        lines.append("")
        lines.append(doc.help)
        lines.append("")
        lines.extend(_options_table(doc.options))
        lines.extend(_subcommands_table(doc.subcommands))
    return "\n".join(lines).rstrip("\n") + "\n"


# --- docs/output.md: the JSON event catalog, from the real serializer ----------------


# Specimens are rendered at a fixed instant, then the local-time `time` value is
# replaced with this canonical stamp so output is byte-identical on any machine.
_SPECIMEN_TIME = "2026-01-01T18:00:00+00:00"
_SPECIMEN_EPOCH = 1_767_290_400.0

_BOOT_SCOPE = ev.ScopeId(kind=ev.ScopeKind.BOOT_STEP, serial=1)
_ENTRY_SCOPE = ev.ScopeId(kind=ev.ScopeKind.ENTRY, serial=2)
_WAIT_SCOPE = ev.ScopeId(kind=ev.ScopeKind.WAIT_REGION, serial=3)


# What each wire event means, keyed by its serialized name. The catalog build
# fails when an on-wire event has no entry here (or an entry goes stale), so a
# new event type cannot ship undocumented.
EVENT_DESCRIPTIONS: dict[str, str] = {
    "run_started": "The process banner: the installed version and the resolved data directory. "
    "The first event of every invocation.",
    "cycle_started": "Scheduled mode only: a new cycle begins (`number` counts from 1).",
    "boot_step_finished": "One preflight step finished; `outcome` is `ok`, `warned`, or `failed`, "
    "with an optional `detail`.",
    "boot_ready": "Preflight finished; the scan begins.",
    "scan_started": "The per-arr scan opened: which arr, and how many library titles it will walk.",
    "item_started": "The scan moved to the next library title (`index` of `total`).",
    "diagnostic": "A problem or notice at any level; `origin` names the subsystem. A position-free "
    "diagnostic carries `during` plus `placed: frontier` (its position is the frontier's best guess), "
    "and one with a captured traceback carries it in `exc`.",
    "scope_opened": "A nested scope opened (`kind` + `serial` identify it); events inside carry a "
    "human-readable `path` breadcrumb instead of the id.",
    "entry_header": "A SeaDex entry block opened for the current title; `message` is the entry state word.",
    "entry_detail": "A labeled line inside an entry block; `message` is `label: value`.",
    "ledger_row": "A self-contained one-line result for a title: `message` is the state word, "
    "`title` the library title.",
    "release_skipped": "A release was skipped at add time; `reason` is `private_only`, "
    "`unsupported_tracker`, or `tracker_not_selected`.",
    "grab_failed": "Adding a release to qBittorrent failed; the title is retried next run.",
    "grab_action": "The grab decision for a title; `message` distinguishes a real add, a dry-run "
    "would-add, and already-downloading.",
    "scope_closed": "The matching close of a `scope_opened` (anything nested deeper closes with it).",
    "cap_reached": "The `advanced.max_torrents_to_add` cap was reached; the run adds nothing further.",
    "scan_finished": "The per-arr scan closed (a boundary event; the summary carries the facts).",
    "run_summary": "The end-of-run scoreboard: the tally counters, plus `needs_action_records` and "
    "`added_records` arrays mirroring the summary's per-title lines.",
    "wait_started": "The wait-for-completion pass opened, watching `total` torrents.",
    "torrent_graduated": "One watched torrent reached a terminal outcome; `message` is the outcome word.",
    "wait_finished": "The wait pass closed, with its imported/deferred/failed tally.",
    "run_finished": "The per-arr run closed (a boundary event).",
    "next_run_scheduled": "Scheduled mode only: when the next cycle fires.",
}

# Event types that never reach the JSON stream, with the reason readers need.
# The catalog build fails if the serializer's actual drop set drifts from this.
JSON_SILENT: dict[str, str] = {
    "BootStepStarted": "a live-cockpit affordance; `boot_step_finished` carries the step's facts",
    "BootStepProgressed": "ephemeral progress for the live cockpit only",
    "BootStepSlow": "a text-surface heads-up; the JSON stream sees the step finish",
    "WaitProgress": "the per-poll wait snapshot; on the JSON stream, wait progress is the "
    "`wait_started` / `torrent_graduated` / `wait_finished` sequence",
}


def _specimen_stream() -> tuple[ev.Event, ...]:
    """A realistic single-run event sequence containing every union member once.

    Ordered like a real run so breadcrumb `path` values are authentic; values
    are fixtures (fixed version, dir, titles), never environment-derived.
    """

    tally = ev.RunTally(
        checked=42,
        added=(
            ev.GrabFact(
                title="Sousou no Frieren",
                coverage="S01",
                url="https://releases.moe/154587/",
                name="[SubsPlease] Sousou no Frieren",
                group="SubsPlease",
            ),
        ),
        up_to_date=31,
        cached=6,
        no_seadex_entry=2,
        seadex_unreachable=0,
        no_releases=1,
        no_mappings=1,
        needs_action=(
            ev.NeedsActionFact(
                title="Made in Abyss",
                coverage="S02",
                group="Vodes",
                url="https://releases.moe/108/",
                reason="private-only release; no public alternative covers these files",
                cause=ev.NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
            ),
        ),
        unmonitored=0,
        queued=0,
        importing=0,
        imported=1,
    )
    summary = ev.RunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        added_count=1,
        tally=tally,
        wait_mode_on=True,
        warnings=1,
        errors=0,
        elapsed_s=63.4,
        tip=ev.NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    )
    return (
        ev.RunStarted(version="v1.0.0", data_dir="/home/user/.local/share/pearlarr"),
        ev.CycleStarted(number=1),
        ev.BootStepStarted(scope=_BOOT_SCOPE, label="Connecting to Sonarr"),
        ev.BootStepProgressed(scope=_BOOT_SCOPE, fraction=0.5),
        ev.BootStepSlow(scope=_BOOT_SCOPE, label="Connecting to Sonarr"),
        ev.BootStepFinished(
            scope=_BOOT_SCOPE,
            label="Connecting to Sonarr",
            outcome=OutcomeCategory.SUCCESS,
            detail="42 series",
            elapsed_s=1.2,
        ),
        ev.BootReady(elapsed_s=3.5),
        ev.ScanStarted(arr=Arr.SONARR, total=42),
        ev.ItemStarted(arr=Arr.SONARR, index=3, total=42, title="Sousou no Frieren"),
        ev.Diagnostic(
            severity=ev.Severity.WARNING,
            message="AniList rate limited - waiting 30s before retrying",
            origin="anilist",
        ),
        ev.ScopeOpened(scope=_ENTRY_SCOPE, label="Sousou no Frieren"),
        ev.EntryHeader(
            state=EntryState.CHECKING,
            title="Sousou no Frieren",
            al_id=154587,
            coverage="S01",
            url="https://releases.moe/154587/",
            scope=_ENTRY_SCOPE,
        ),
        ev.EntryDetail(
            label="status",
            value=ev.StyledValue("missing episodes"),
            tail="S01E27-E28",
            scope=_ENTRY_SCOPE,
        ),
        ev.LedgerRow(state=EntryState.UNCHANGED, label="Sousou no Frieren", scope=_ENTRY_SCOPE),
        ev.ReleaseSkipped(
            group="Vodes",
            tracker="AnimeBytes",
            reason=ev.SkipReason.PRIVATE_ONLY,
            url="https://releases.moe/108/",
            scope=_ENTRY_SCOPE,
        ),
        ev.GrabFailed(
            group="SubsPlease",
            url="https://nyaa.si/view/1734567",
            error="qBittorrent rejected the torrent",
            scope=_ENTRY_SCOPE,
        ),
        ev.GrabAction(
            status=ev.GrabStatus.ADDING,
            groups=(ev.RecommendedGroup(name="SubsPlease", tags=("best",)),),
            added=(ev.ReleaseName(name="[SubsPlease] Sousou no Frieren", group="SubsPlease"),),
            downloading=(),
            waiting_to_import=True,
            scope=_ENTRY_SCOPE,
        ),
        ev.ScopeClosed(scope=_ENTRY_SCOPE),
        ev.CapReached(cap=25),
        ev.ScanFinished(arr=Arr.SONARR),
        ev.RunSummaryReady(summary=summary),
        ev.ScopeOpened(scope=_WAIT_SCOPE, label="wait"),
        ev.WaitStarted(total=1, pulse_s=300.0, scope=_WAIT_SCOPE),
        ev.WaitProgress(snapshot=ev.WaitSnapshot(torrents=(), elapsed_s=30.0), scope=_WAIT_SCOPE),
        ev.TorrentGraduated(
            label="[SubsPlease] Sousou no Frieren",
            outcome=Outcome.IMPORTED,
            files=28,
            waited_s=412.0,
            scope=_WAIT_SCOPE,
        ),
        ev.WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=430.0, scope=_WAIT_SCOPE),
        ev.ScopeClosed(scope=_WAIT_SCOPE),
        ev.RunFinished(arr=Arr.SONARR),
        ev.NextRunScheduled(at=datetime(2026, 1, 2, 0, 0, tzinfo=UTC)),
    )


@dataclass(frozen=True)
class EventSpecimen:
    """One wire event: its serialized name, meaning, and pretty-printed payload."""

    name: str
    description: str
    payload: str


def build_event_catalog() -> tuple[EventSpecimen, ...]:
    """Run the specimen stream through the real JSON sink, one specimen per event name.

    Hard-fails when the stream misses a union member, an on-wire event lacks a
    description, a description goes stale, or the serializer's silent set
    drifts from `JSON_SILENT` - so the catalog cannot quietly lose coverage.
    """

    stream = _specimen_stream()
    members = {member.__name__ for member in cast("tuple[type[object], ...]", get_args(ev.Event.__value__))}
    streamed = {type(event).__name__ for event in stream}
    if members - streamed:
        raise GenerationError(f"specimen stream misses event types: {', '.join(sorted(members - streamed))}")

    buffer = io.StringIO()
    renderer = JsonRenderer(buffer)
    renderer.set_level(int(ev.Severity.DEBUG))
    specimens: list[EventSpecimen] = []
    seen: set[str] = set()
    silent: set[str] = set()
    emitted = 0
    for event in stream:
        renderer.handle(event, _SPECIMEN_EPOCH)
        lines = buffer.getvalue().splitlines()
        if len(lines) == emitted:
            silent.add(type(event).__name__)
            continue
        for raw in lines[emitted:]:
            parsed: object = loads(raw)
            if not is_json_obj(parsed):
                raise GenerationError(f"the JSON sink emitted a non-object line: {raw[:60]!r}")
            payload = dict(parsed)
            payload["time"] = _SPECIMEN_TIME
            name = payload.get("event")
            if not isinstance(name, str):
                raise GenerationError(f"a JSON line carries no event name: {raw[:60]!r}")
            if name in seen:
                continue
            seen.add(name)
            if name not in EVENT_DESCRIPTIONS:
                raise GenerationError(f"wire event {name} has no entry in EVENT_DESCRIPTIONS")
            specimens.append(
                EventSpecimen(
                    name=name,
                    description=EVENT_DESCRIPTIONS[name],
                    payload=dumps(payload, indent=2, ensure_ascii=False),
                ),
            )
        emitted = len(lines)

    if stale := set(EVENT_DESCRIPTIONS) - seen:
        raise GenerationError(f"EVENT_DESCRIPTIONS entries match no wire event: {', '.join(sorted(stale))}")
    if silent != set(JSON_SILENT):
        raise GenerationError(
            f"the serializer's silent set drifted: observed {sorted(silent)}, documented {sorted(JSON_SILENT)}",
        )
    return tuple(specimens)


def render_event_catalog() -> str:
    """The event-catalog island: one specimen per wire event, plus the silent list."""

    lines = [
        "Specimens are generated by running one representative event of every kind",
        "through the real JSON serializer, so serialized names, enum values, and",
        "null-vs-absent behavior are the wire truth. The stream writes each object",
        "on a single line; specimens are pretty-printed here, with `time` normalized.",
        "",
    ]
    for specimen in build_event_catalog():
        lines.extend(
            (
                f"### `{specimen.name}`",
                "",
                specimen.description,
                "",
                "```json",
                *specimen.payload.splitlines(),
                "```",
                "",
            ),
        )
    lines.extend(("### Events that never reach the JSON stream", ""))
    lines.extend(f"- `{name}` - {reason}." for name, reason in JSON_SILENT.items())
    return "\n".join(lines) + "\n"


def render_output() -> str:
    """docs/output.md with its event-catalog island refreshed."""

    if not OUTPUT_PATH.exists():
        raise GenerationError(f"{OUTPUT_PATH} does not exist; author its prose skeleton first")
    document = OUTPUT_PATH.read_text(encoding="utf-8")
    return stitch(document, "json-events", render_event_catalog(), EVENTS_SOURCE)


def artifacts() -> dict[Path, str]:
    """Every generated artifact, rendered."""

    groups = build_groups()
    return {
        SAMPLE_PATH: render_sample(groups),
        SCHEMA_PATH: render_schema(groups),
        CONFIGURATION_PATH: render_configuration(groups),
        CONTRIBUTING_PATH: render_contributing(),
        ARCHITECTURE_PATH: render_architecture(),
        CLI_PATH: render_cli(),
        OUTPUT_PATH: render_output(),
    }


def main() -> int:
    """Entry point: write the artifacts, or verify them with `--check`."""

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
