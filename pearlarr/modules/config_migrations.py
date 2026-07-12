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

from collections.abc import Callable, Mapping
from dataclasses import dataclass

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
