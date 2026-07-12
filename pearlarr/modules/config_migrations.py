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

# The v0-era coalesce targets, spelled as the historical literals.
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_IMPORT_MODES = ("auto", "move", "copy")


@dataclass(frozen=True)
class MigrationOutcome:
    """What one migration pass did: the version it found and the functional changes.

    `notes` is empty when the pass only stamped `config_version` (the mapping's
    keys and values were already readable as-is).
    """

    from_version: int
    notes: tuple[str, ...]


def declared_version(config: Mapping[str, Json]) -> int | None:
    """The `config_version` a raw mapping declares.

    Absent or blank counts as 0 (a pre-versioning file); a non-int value is
    None, so the chain stays away and validation reports the bad key itself.
    """

    value = config.get("config_version")
    if value is None:
        return 0
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _group(config: dict[str, Json], key: str) -> dict[str, Json] | None:
    """A settings group as a mutable mapping, or None when absent or not a mapping."""

    value = config.get(key)
    return value if is_json_obj(value) else None


def _to_v1(config: dict[str, Json]) -> list[str]:
    """v0 -> v1: fold the pre-versioning keys and values that 1.0.0 removed or narrowed."""

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
    if (advanced := _group(config, "advanced")) is not None:
        level = advanced.get("log_level")
        if isinstance(level, str) and level.upper() not in _LOG_LEVELS:
            # Free-form once: an unknown level warned and ran at INFO; keep that
            # instead of newly rejecting the file.
            advanced.pop("log_level")
            notes.append(f"advanced.log_level {level!r} is not a log level - dropped (runs at INFO)")
    if (imports := _group(config, "imports")) is not None:
        mode = imports.get("mode")
        if isinstance(mode, str) and mode not in _IMPORT_MODES:
            # Free-form once (forwarded verbatim to Sonarr, which refused it at
            # import time); auto is the closest working reading.
            imports.pop("mode")
            notes.append(f"imports.mode {mode!r} is not auto/move/copy - dropped (auto)")
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
# match neither and pass through untouched.
_TOP_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")
_FIELD_KEY = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):")


def _value_lines(indent: str, key: str, value: Json) -> list[str]:
    """`key: value` at `indent`; a filled container goes block-style below the key."""

    if value is None:
        return [f"{indent}{key}:"]
    if isinstance(value, (dict, list)) and value:
        dumped = yaml.safe_dump(value, default_flow_style=False, sort_keys=False).rstrip("\n")
        return [f"{indent}{key}:", *(f"{indent}  {line}" for line in dumped.splitlines())]
    scalar = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).partition("\n")[0]
    return [f"{indent}{key}: {scalar}"]


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
