"""Config schema versioning: the chain that brings an older config file forward.

`CONFIG_VERSION` names the current schema; `AppConfig.load` runs
`migrate_mapping` over the raw parsed YAML before validation, so a config
written for an older Pearlarr keeps loading (in memory - the file on disk is
never touched by a load). `pearlarr config migrate` rewrites the file itself,
via `render_migrated_config`.

Migration steps are frozen history: they spell old keys and values as string
literals - never live enums or constants, which move on with the schema - and
each step brings a mapping exactly one version forward.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import yaml

from .json_narrow import is_json_obj
from .seadex_types import Json

CONFIG_VERSION = 1

# The one remediation sentence, shared by every surface that reports an
# old-schema file (and quoted by docs/troubleshooting.md's anchor grep).
MIGRATE_HINT = "run pearlarr config migrate to update the file (a backup is kept)"


@dataclass(frozen=True)
class MigrationOutcome:
    """What one migration pass did: the version it found and the functional changes."""

    from_version: int
    """The schema version the mapping declared before this pass ran."""

    notes: tuple[str, ...]
    """Human-readable functional changes; empty when the pass only stamped config_version."""


def declared_version(config: Mapping[str, Json]) -> int | None:
    """The `config_version` a raw mapping declares.

    An absent key counts as 0 (a pre-versioning file). Everything else that is
    not a non-negative int - a blank key (which takes the field default, like
    every blank key), a bool, a string, a negative - is None: the chain stays
    away, and validation speaks for the key itself.
    """

    if "config_version" not in config:
        return 0
    value = config["config_version"]
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _group(config: dict[str, Json], key: str) -> dict[str, Json] | None:
    """A settings group as a mutable mapping, or None when absent or not a mapping."""

    value = config.get(key)
    return value if is_json_obj(value) else None


def _to_v1(config: dict[str, Json]) -> list[str]:
    """v0 -> v1: fold the pre-versioning schema that 1.0.0 removed.

    Only removed KEYS and removed ENUM VALUES fold - spellings that were once
    part of the schema and cannot be a typo against the current one. A value
    that was never valid (a misspelled log level, an unknown import mode) is
    left for validation to reject by name: a version-less file may just as
    well be a new hand-written config, and typos must stay loud.
    """

    notes: list[str] = []
    if (seadex := _group(config, "seadex")) is not None:
        if "public_only" in seadex:
            # Replaced by private_releases; both old values behaved as today's
            # warn (private releases were never grabbed), so dropping the key
            # keeps the behavior.
            seadex.pop("public_only")
            notes.append(
                "seadex.public_only was replaced by seadex.private_releases - dropped (behavior unchanged)",
            )
        if seadex.get("private_releases") == "allow":
            seadex["private_releases"] = "warn"
            notes.append(
                "seadex.private_releases 'allow' was removed - folded to 'warn' (private releases were never grabbed)",
            )
    return notes


@dataclass(frozen=True)
class _Migration:
    """One schema step: brings a mapping from `to_version - 1` to `to_version`."""

    to_version: int
    apply: Callable[[dict[str, Json]], list[str]]


_MIGRATIONS: tuple[_Migration, ...] = (_Migration(to_version=1, apply=_to_v1),)


def migrate_mapping(config: dict[str, Json]) -> MigrationOutcome | None:
    """Bring a raw config mapping to `CONFIG_VERSION`, in place.

    Returns what happened, or None when the mapping was already current - or
    when its version key is newer or unusable, which the chain leaves alone so
    validation can refuse it by name.
    """

    version = declared_version(config)
    if version is None or version >= CONFIG_VERSION:
        return None
    notes: list[str] = []
    for migration in _MIGRATIONS:
        if migration.to_version > version:
            notes.extend(migration.apply(config))
    config["config_version"] = CONFIG_VERSION
    return MigrationOutcome(from_version=version, notes=tuple(notes))


# The template's two key shapes: a top-level key (group header or scalar) and a
# group field at the generator's fixed two-space indent. Comment and blank lines
# match neither and pass through untouched. A pin test asserts every template
# line is one of those four shapes, so a generator format change fails CI
# instead of silently un-matching the splice.
_TOP_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")
_FIELD_KEY = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):")

# yaml's line-folding knob: without this, safe_dump wraps a long plain scalar
# onto continuation lines and a single-line splice would truncate the value.
_NO_WRAP = 2**20


def _scalar_token(value: Json) -> str:
    """One YAML token that round-trips `value` on a single physical line.

    Plain style for readability; a string carrying line breaks or tabs is
    forced into double-quoted style, whose escapes keep it on one line.
    """

    style = '"' if isinstance(value, str) and any(ch in value for ch in "\n\r\t\x85") else None
    dumped = yaml.safe_dump(value, default_flow_style=True, width=_NO_WRAP, default_style=style)
    return dumped.partition("\n")[0]


def _value_lines(indent: str, key: str, value: Json) -> list[str]:
    """`key: value` at `indent`; a filled container goes block-style below the key."""

    if value is None:
        return [f"{indent}{key}:"]
    if isinstance(value, (dict, list)) and value:
        dumped = yaml.safe_dump(value, default_flow_style=False, sort_keys=False, width=_NO_WRAP).rstrip("\n")
        return [f"{indent}{key}:", *(f"{indent}  {line}" for line in dumped.splitlines())]
    return [f"{indent}{key}: {_scalar_token(value)}"]


def render_migrated_config(template: str, config: Mapping[str, Json]) -> str:
    """The current annotated template with a migrated mapping's values spliced in.

    Comments, docs and key order come from the template (what `config init`
    ships); a key the mapping sets explicitly takes the mapping's value, every
    other line keeps the template's, so blank-means-default stays blank. The
    mapping must already be migrated and validated: every key it carries exists
    in the template, so nothing can be dropped.
    """

    out: list[str] = []
    group: dict[str, Json] | None = None
    for line in template.splitlines():
        if top := _TOP_KEY.match(line):
            key = top.group(1)
            value = config.get(key)
            if is_json_obj(value):
                group = value
                out.append(line)
            else:
                group = None
                out.extend(_value_lines("", key, value) if key in config else [line])
        elif (field := _FIELD_KEY.match(line)) and group is not None and field.group(1) in group:
            out.extend(_value_lines("  ", field.group(1), group[field.group(1)]))
        else:
            out.append(line)
    return "\n".join(out) + "\n"
