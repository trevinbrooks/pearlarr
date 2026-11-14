"""Config schema versioning: the chain that brings an older config file forward.

`CONFIG_VERSION` names the current schema. `AppConfig.load` runs
`migrate_mapping` over the raw parsed YAML before validation, so a config
written for an older Pearlarr keeps loading (in memory - the file on disk is
never touched by a load). The run path and `pearlarr config migrate` then
rewrite the file itself, via `render_migrated_config`.

Migration steps are frozen history: they spell old keys and values as string
literals - never live enums or constants, which move on with the schema - and
each step brings a mapping exactly one version forward.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import yaml

from .boundary import Json
from .json_narrow import is_json_obj

CONFIG_VERSION = 7

# The one remediation sentence, shared by every surface that reports an
# old-schema file (and quoted by docs/troubleshooting.md's anchor grep).
MIGRATE_HINT = "run pearlarr config migrate to update the file (a backup is kept)"


@dataclass(frozen=True)
class MigrationOutcome:
    """What one migration pass did: the version it found and the functional changes."""

    from_version: int
    """The schema version the mapping declared before this pass ran."""

    notes: tuple[str, ...]
    """Human-readable functional changes. Empty when the pass only bumped config_version."""


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


def _group_to_fill(config: dict[str, Json], key: str) -> dict[str, Json] | None:
    """A settings group to move keys into: the existing one, a new one when absent or blank, else None.

    A malformed group (not a mapping) is left for validation to reject by name.
    """

    group: dict[str, Json] | None = _group(config, key)
    if group is None and config.get(key) is None:
        group = {}
        config[key] = group
    return group


def _to_v1(config: dict[str, Json]) -> list[str]:
    """v0 -> v1: convert the pre-versioning schema that 1.0.0 removed.

    Only removed KEYS and removed ENUM VALUES convert - spellings that were once
    part of the schema and cannot be a typo against the current one. A value
    that was never valid (a misspelled log level, an unknown import mode) is
    left for validation to reject by name: a version-less file may just as
    well be a new hand-written config, and typos must stay loud.
    """

    notes: list[str] = []
    if (seadex := _group(config, "seadex")) is not None:
        if "public_only" in seadex:
            # Replaced by private_releases. Both old values behaved as today's
            # warn (private releases were never grabbed), so dropping the key
            # keeps the behavior.
            seadex.pop("public_only")
            notes.append(
                "seadex.public_only was replaced by seadex.private_releases - dropped (behavior unchanged)",
            )
        if seadex.get("private_releases") == "allow":
            seadex["private_releases"] = "warn"
            notes.append(
                "seadex.private_releases 'allow' was removed - converted to 'warn' (private releases were never grabbed)",
            )
    return notes


def _to_v2(config: dict[str, Json]) -> list[str]:
    """v1 -> v2: `imports.post_import_category` became per-arr; `wait_mode` defaults on.

    The old single category key covered both arrs, so its value lands on both
    groups (inert on an unconfigured arr). A group that is absent or blank is
    created to carry it; a malformed group is left for validation to reject by
    name. The `wait_mode` note is informational: a file that relied on the old
    `off` default now gets `blocking`, and only an explicit `off` disables.
    """

    notes: list[str] = []
    imports = _group(config, "imports")
    if imports is not None and "post_import_category" in imports:
        category = imports.pop("post_import_category")
        for arr in ("sonarr", "radarr"):
            if (group := _group_to_fill(config, arr)) is not None:
                group["post_import_category"] = category
        notes.append(
            "imports.post_import_category is now per-arr - moved to sonarr.post_import_category "
            "and radarr.post_import_category (same value, same behavior)",
        )
    if imports is None or imports.get("wait_mode") is None:
        notes.append(
            "imports.wait_mode now defaults to blocking (wait for downloads, then import them) - "
            "set it to off to keep the previous disabled behavior",
        )
    return notes


def _to_v3(config: dict[str, Json]) -> list[str]:
    """v2 -> v3: an omitted per-arr category now adopts the arr's own download-client category.

    Note-only, like the v2 `wait_mode` notice: nothing is rewritten. The note
    fires unconditionally - whether the flip bites depends on connection
    state a file mapping cannot see (env overrides, qBittorrent credentials
    added after the version was written).
    """

    return [
        (
            "an omitted torrent_category / post_import_category now adopts the matching category of that "
            "arr's own qBittorrent download client - an explicit value overrides it, and an explicit "
            'blank ("") keeps no category at all'
        ),
    ]


def _to_v4(config: dict[str, Json]) -> list[str]:
    """v3 -> v4: drop the removed `imports.pending_max_age_days`, noting it only when the file set it."""

    imports = _group(config, "imports")
    if imports is None or "pending_max_age_days" not in imports:
        return []
    imports.pop("pending_max_age_days")
    return [
        (
            "imports.pending_max_age_days was removed - dropped "
            "(a pending import now stays tracked until it imports or leaves qBittorrent)"
        ),
    ]


def _to_v5(config: dict[str, Json]) -> list[str]:
    """v4 -> v5: drop the removed `sonarr.ignore_movies_in_radarr`, saying what changes when it was on.

    A blank or a false value was off. Anything else gets the note about the
    change. A quoted "true" loaded as on, and a quoted "false" getting the note
    too is harmless.
    """

    sonarr = _group(config, "sonarr")
    if sonarr is None or "ignore_movies_in_radarr" not in sonarr:
        return []
    if sonarr.pop("ignore_movies_in_radarr") in (None, False):
        return ["sonarr.ignore_movies_in_radarr was removed - dropped (it was off, so nothing changes)"]
    return [
        (
            "sonarr.ignore_movies_in_radarr was removed - dropped (Sonarr now checks a movie Radarr also has, "
            "so it can be grabbed for Sonarr too. To ignore one, unmonitor its episode in Sonarr and turn on "
            "sonarr.ignore_unmonitored)"
        ),
    ]


def _to_v6(config: dict[str, Json]) -> list[str]:
    """v5 -> v6: drop the removed `seadex.use_torrent_hash_to_filter`, saying what changes when it was on.

    A blank or `false` was off. Any other value gets the note about what changes, as in `_to_v5`.
    """

    seadex = _group(config, "seadex")
    if seadex is None or "use_torrent_hash_to_filter" not in seadex:
        return []
    if seadex.pop("use_torrent_hash_to_filter") in (None, False):
        return ["seadex.use_torrent_hash_to_filter was removed - dropped (it was off, so nothing changes)"]
    return [
        (
            "seadex.use_torrent_hash_to_filter was removed - dropped (a release is now compared with the files "
            "the arr has, by release group and file size, so one whose files are gone or changed is grabbed "
            "again even if Pearlarr grabbed it before. To ignore a title, add its AniList ID to "
            "seadex.ignore_anilist_ids)"
        ),
    ]


@dataclass(frozen=True)
class ConfigKey:
    """A config key by its group and name."""

    group: str
    key: str


V7_KEY_RENAMES: Final[Mapping[ConfigKey, ConfigKey]] = {
    ConfigKey("imports", "wait_timeout"): ConfigKey("imports", "download_timeout"),
    ConfigKey("imports", "ready_timeout"): ConfigKey("imports", "import_timeout"),
    ConfigKey("imports", "digest_interval"): ConfigKey("imports", "progress_line_interval"),
    ConfigKey("imports", "mode"): ConfigKey("imports", "import_mode"),
    ConfigKey("advanced", "cache_time"): ConfigKey("mappings", "refresh_days"),
    ConfigKey("mappings", "anime_mappings"): ConfigKey("mappings", "anime_ids"),
    ConfigKey("mappings", "anidb_mappings"): ConfigKey("mappings", "anime_lists"),
    ConfigKey("mappings", "anibridge_mappings"): ConfigKey("mappings", "anibridge"),
}
"""The key each version 7 key replaced, old to new."""


def _to_v7(config: dict[str, Json]) -> list[str]:
    """v6 -> v7: rename the keys in `V7_KEY_RENAMES`, and rewrite `imports.wait_mode: hybrid` to `blocking`.

    A key moves into its new group, created when absent or blank (as in `_to_v2`). When the new
    key is already set, the old one stays and validation rejects it as unknown.
    """

    renamed: list[str] = []
    for old, new in V7_KEY_RENAMES.items():
        source = _group(config, old.group)
        if source is None or old.key not in source:
            continue
        target = _group_to_fill(config, new.group)
        if target is None or new.key in target:
            continue
        target[new.key] = source.pop(old.key)
        renamed.append(f"{old.group}.{old.key} is now {new.group}.{new.key}")
    notes: list[str] = []
    if renamed:
        notes.append(f"config keys were renamed - {'; '.join(renamed)} (same values, same behavior)")
    imports = _group(config, "imports")
    if imports is not None and imports.get("wait_mode") == "hybrid":
        imports["wait_mode"] = "blocking"
        notes.append("imports.wait_mode hybrid was removed - rewritten to blocking (the same behavior)")
    return notes


@dataclass(frozen=True)
class _Migration:
    """One schema step: brings a mapping from `to_version - 1` to `to_version`."""

    to_version: int
    apply: Callable[[dict[str, Json]], list[str]]


_MIGRATIONS: tuple[_Migration, ...] = (
    _Migration(to_version=1, apply=_to_v1),
    _Migration(to_version=2, apply=_to_v2),
    _Migration(to_version=3, apply=_to_v3),
    _Migration(to_version=4, apply=_to_v4),
    _Migration(to_version=5, apply=_to_v5),
    _Migration(to_version=6, apply=_to_v6),
    _Migration(to_version=7, apply=_to_v7),
)


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

    Plain style for readability. A string carrying line breaks or tabs is
    forced into double-quoted style, whose escapes keep it on one line.
    """

    style = '"' if isinstance(value, str) and any(ch in value for ch in "\n\r\t\x85") else None
    dumped = yaml.safe_dump(value, default_flow_style=True, width=_NO_WRAP, default_style=style)
    return dumped.partition("\n")[0]


def _value_lines(indent: str, key: str, value: Json) -> list[str]:
    """`key: value` at `indent`. A filled container goes block-style below the key."""

    if value is None:
        return [f"{indent}{key}:"]
    if isinstance(value, (dict, list)) and value:
        dumped = yaml.safe_dump(value, default_flow_style=False, sort_keys=False, width=_NO_WRAP).rstrip("\n")
        return [f"{indent}{key}:", *(f"{indent}  {line}" for line in dumped.splitlines())]
    return [f"{indent}{key}: {_scalar_token(value)}"]


def render_migrated_config(template: str, config: Mapping[str, Json]) -> str:
    """The current annotated template with a migrated mapping's values spliced in.

    Comments, docs and key order come from the template (what `config init`
    ships). A key the mapping sets explicitly takes the mapping's value, every
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
        elif (field := _FIELD_KEY.match(line)) and group is not None and (fkey := field.group(1)) in group:
            out.extend(_value_lines("  ", fkey, group[fkey]))
        else:
            out.append(line)
    return "\n".join(out) + "\n"
