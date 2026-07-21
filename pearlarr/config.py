"""Application configuration: typed, validated settings loaded from the YAML file.

`AppConfig` is a Pydantic model tree over the config file. Pydantic owns parsing,
defaults, coercion (including the present-but-blank-YAML-key footgun) and validation:
an unknown/typo'd key or a bad value raises a clear `ValidationError` instead of
being silently dropped or coalesced. The rest of the package reads
`config.seadex.want_best` rather than `config.get("want_best", True)`.

Each settings group is its own submodel (`sonarr`/`radarr`/`qbittorrent`/
`seadex`/`imports`/`notifications`/`advanced`/`mappings`). The connection
`url`/`api_key` for each arr are optional at parse time and enforced lazily at
point-of-use via `AppConfig.require_connection`, so a config filled in for only
one arr still validates and runs.
"""

import contextlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import md5, sha256
from types import UnionType
from typing import Any, Literal, NamedTuple, Union, cast, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from .config_migrations import (
    CONFIG_VERSION,
    MigrationOutcome,
    migrate_mapping,
    render_migrated_config,
)
from .env_registry import ENV_CONFIG_DELIMITER, ENV_CONFIG_PREFIX
from .json_narrow import is_json_obj
from .manual_import import ImportWaitMode
from .seadex_types import Json

# Tracker name classification. The *_TRACKER_NAMES tuples keep SeaDex's exact casings
# (the docs generator renders them into the sample and reference). The sets are
# casefolded so membership tests match the casefolded `seadex.trackers` setting and
# the casefolded tracker names from SeaDex.
PUBLIC_TRACKER_NAMES: tuple[str, ...] = ("Nyaa", "AnimeTosho", "AniDex", "RuTracker")
PRIVATE_TRACKER_NAMES: tuple[str, ...] = (
    "AB",
    "BeyondHD",
    "PassThePopcorn",
    "BroadcastTheNet",
    "HDBits",
    "Blutopia",
    "Aither",
)
OTHER_TRACKER_NAMES: tuple[str, ...] = ("Other", "OtherPrivate")

PUBLIC_TRACKERS = {tracker.casefold() for tracker in PUBLIC_TRACKER_NAMES}
PRIVATE_TRACKERS = {tracker.casefold() for tracker in PRIVATE_TRACKER_NAMES}

# Every name the seadex `Tracker` enum can emit (casefolded), so the cli can warn on
# configured trackers that match nothing. Literals keep this module's imports light.
# A parity test pins them against the installed enum.
KNOWN_TRACKERS = PUBLIC_TRACKERS | PRIVATE_TRACKERS | {tracker.casefold() for tracker in OTHER_TRACKER_NAMES}

CONFIG_TEMPLATE_FILE = "config_sample.yml"


class Arr(StrEnum):
    """Which *arr the run targets.

    A `StrEnum` so the value still equals its string (`Arr.SONARR == "sonarr"`),
    serializes as a bare JSON cache key, and selects the matching config group -
    while making the two valid arrs the only representable states.
    """

    SONARR = "sonarr"
    RADARR = "radarr"


class ArrTarget(NamedTuple):
    """One arr-selection target: which arr, plus the optional single item to narrow to.

    `item_id` is a TMDB id for Radarr, a TVDB/series id for Sonarr (None runs the
    whole library). A NamedTuple so it still unpacks and compares as a plain
    `(arr, item_id)` pair, keeping the name that says what the second slot means.
    """

    arr: Arr
    item_id: int | None = None


class PrivateReleaseAction(StrEnum):
    """The `seadex.private_releases` policy for private-tracker releases.

    Private releases are never grabbed: SeaDex carries no downloadable link or
    infohash for them, and no private-tracker auth is supported. The policy
    decides what happens when a title's preferred release is private-only.
    """

    WARN = "warn"
    """Warn and skip without caching, so the title is re-checked every run until a public release appears."""

    FALLBACK = "fallback"
    """Grab the entry's best public alternative instead. Warn only when none exists at all."""


def template_path() -> str:
    """Absolute path to the bundled config template shipped beside this module."""

    return os.path.join(os.path.dirname(__file__), CONFIG_TEMPLATE_FILE)


def starter_template_text() -> str:
    """The bundled template minus its generated-file banner (the form users own)."""

    with open(template_path(), encoding="utf-8") as f:
        return "".join(line for line in f if not line.startswith("# GENERATED"))


def write_starter_config(path: str) -> None:
    """Copy the bundled template to `path` as a user-owned starter config.

    Drops the template's generated-file banner (the copy is now yours to
    edit, not a generated artifact) and restricts the copy to owner-only.
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(starter_template_text())
    restrict_config_permissions(path)


def restrict_config_permissions(path: str) -> None:
    """Best-effort `chmod 0600`: the config carries plaintext API keys.

    Both creation paths (the first-run template copy and `config init`) call
    this so a fresh config never lands group/other-readable. Best-effort because
    a filesystem that doesn't support modes (or a race with a deleted file) must
    not turn the write into a crash.
    """

    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def config_permissions_loose(path: str) -> bool:
    """True when the config file is accessible to group/other (POSIX only).

    The load path warns on this: an existing config predating the 0600-on-create
    hardening may still expose its credentials. Always False on non-POSIX
    (Windows ACLs don't map onto the mode bits) and on an unstatable path.
    """

    if os.name != "posix":
        return False
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return False
    return bool(mode & 0o077)


def secret_value(secret: SecretStr | None) -> str | None:
    """Unwrap an optional `SecretStr` at a point of use."""

    return secret.get_secret_value() if secret is not None else None


def strip_userinfo(value: str) -> str:
    """Mask a `user:pass@` login embedded in a URL/host value.

    Every user-facing rendering of a configured URL (`config show`, arr
    connection errors) goes through this: the login is a credential, and the
    host is what should stay visible.
    """

    scheme, sep, rest = value.partition("://")
    if not sep:
        scheme, rest = "", value
    authority, slash, tail = rest.partition("/")
    if "@" not in authority:
        return value
    prefix = f"{scheme}://" if sep else ""
    return f"{prefix}REDACTED@{authority.rpartition('@')[2]}{slash}{tail}"


class _ConfigBase(BaseModel):
    """Shared base for every settings group: strict, immutable, blank-tolerant.

    `extra="forbid"` turns a typo'd key into a `ValidationError` (the whole point
    of validating). `frozen=True` matches the project's value-object convention and
    makes sharing one loaded config across both arrs provably safe.
    `hide_input_in_errors=True` keeps a rejected value (which could be a
    credential pasted under the wrong key) out of the ValidationError text.
    `use_attribute_docstrings=True` lifts each field's attribute docstring into
    `FieldInfo.description` - the one authored home the docs generator renders.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_none(cls, data: Any) -> Any:
        """Let blank YAML keys fall back to their declared default.

        A present-but-blank YAML key (`foo:`) parses to `None`. A plain field
        default would not apply (the key is present) and `None` would be rejected
        by an `int`/`str` field. Drop `None` *only for known fields* so the
        default applies. Unknown keys are kept regardless of blankness, so
        `extra="forbid"` still flags both `typo: value` and a bare `typo:`.
        """

        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, Any]", data)
        fields = cls.model_fields
        return {key: value for key, value in raw.items() if value is not None or key not in fields}


class ArrSettings(_ConfigBase):
    """Connection + per-arr behavior, shared by the `sonarr` and `radarr` groups.

    `api_key` is a `SecretStr` so a dumped/logged model masks it.
    `AppConfig.require_connection` (and the Sonarr->Radarr cross-check)
    unwrap it at the client-construction boundary. `verify_ssl: false` is the
    last-resort escape hatch for a self-signed arr whose CA can't be trusted via
    the OS store / `SSL_CERT_FILE`.
    """

    url: str | None = None
    """Base URL of the instance's web UI, e.g. `http://localhost:8989` for Sonarr or `http://localhost:7878` for Radarr.

    Blank leaves this arr unconfigured and its runs are skipped.
    """

    api_key: SecretStr | None = None
    """API key of the instance, from Settings > General in its web UI.

    Blank leaves this arr unconfigured and its runs are skipped.
    """

    verify_ssl: bool = True
    """Verify the instance's HTTPS certificate.

    Turn off only for a self-signed instance whose CA cannot be trusted via the
    OS store or `SSL_CERT_FILE`.
    """

    ignore_unmonitored: bool = False
    """Skip items the arr does not monitor.

    For Radarr this means unmonitored movies. For Sonarr, unmonitored series or
    entries whose episodes are all unmonitored.
    """

    torrent_category: str | None = None
    """qBittorrent category applied to torrents grabbed for this arr. Blank applies no category."""


class SonarrSettings(ArrSettings):
    """Sonarr adds one cross-arr flag the Radarr group must not accept.

    `ignore_movies_in_radarr` is read only on a Sonarr run. Declaring it here (not
    on the shared base) means `extra="forbid"` correctly rejects it under `radarr`.
    """

    ignore_movies_in_radarr: bool = False
    """Skip movies that already exist in Radarr instead of also grabbing them on the Sonarr side."""


class QbittorrentSettings(_ConfigBase):
    """qBittorrent connection.

    The connection fields are modelled explicitly. `options` is a scoped escape
    hatch for the remaining `qbittorrentapi.Client` kwargs (e.g.
    `VERIFY_WEBUI_CERTIFICATE` for a self-signed-HTTPS WebUI, `REQUESTS_ARGS`) so the
    explicit model doesn't drop connectivity the old free-form `qbit_info` splat allowed.
    """

    host: str | None = None
    """WebUI address of qBittorrent, e.g. `http://localhost:8080`.

    Leave any of `host`, `username`, or `password` blank to run in preview mode:
    runs report what they would grab, but nothing is added.
    """

    username: str | None = None
    """WebUI username. Blank switches to preview mode (see `host`)."""

    password: SecretStr | None = None
    """WebUI password. Blank switches to preview mode (see `host`)."""

    tags: list[str] | None = None
    """Tags applied to every torrent added. Blank applies no tags."""

    # Splatted into qbittorrentapi.Client alongside the connection triple.
    options: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments passed to the qbittorrent-api client.

    For example `VERIFY_WEBUI_CERTIFICATE: false` for a WebUI behind a
    self-signed HTTPS certificate.
    """

    def credentials(self) -> tuple[str, str, str] | None:
        """The `(host, username, password)` triple, or `None` if any part is unset.

        All three are needed to add torrents. A missing one means run in preview mode
        (nothing is grabbed). The caller builds the client with explicit kwargs. The
        password is unwrapped here, at its point of use.
        """

        if self.host and self.username and self.password:
            return self.host, self.username, self.password.get_secret_value()
        return None


class SeadexSettings(_ConfigBase):
    """SeaDex release-selection filters."""

    private_releases: PrivateReleaseAction = PrivateReleaseAction.WARN
    """What to do when a title's preferred release sits only on private trackers.

    Private releases are never grabbed: SeaDex carries no download link for
    them, and no private-tracker auth is supported.
    """

    prefer_dual_audio: bool = True
    """Prefer dual-audio releases. When off, prefer Japanese-audio releases."""

    want_best: bool = True
    """Prefer releases SeaDex marks as best."""

    ignore_tags: list[str] = Field(default_factory=list)
    """SeaDex release tags to skip, e.g. `Dolby Vision` or `Deband Required`. Empty skips none."""

    # Private trackers are in the default even though private releases won't be
    # grabbed: they're filtered later, after the overlap check against what's
    # already downloaded.
    trackers: set[str] = Field(default_factory=lambda: PUBLIC_TRACKERS | PRIVATE_TRACKERS)
    """Tracker names considered during release selection, case-insensitive.

    Empty or absent considers every named tracker except the `Other` and
    `OtherPrivate` catch-alls. A listed tracker is not necessarily grabbable:
    private releases are never downloaded, and grabs come only from the public
    trackers Pearlarr can parse.
    """

    ignore_anilist_ids: set[int] = Field(default_factory=set[int])
    """AniList IDs never processed. Empty ignores none."""

    ignore_seadex_update_times: bool = False
    """Re-check entries even when SeaDex has not updated them since the last check."""

    use_torrent_hash_to_filter: bool = False
    """Match what is already downloaded by torrent hash instead of by release group name."""

    @field_validator("trackers", mode="before")
    @classmethod
    def _casefold_trackers(cls, value: Any) -> Any:
        """Casefold explicit trackers so membership tests match SeaDex's names.

        An absent `trackers` key takes the `PUBLIC | PRIVATE` default_factory. An
        explicit empty list/string coalesces to that same default here (empty means "no
        restriction", never "match nothing" - which would silently grab nothing,
        mirroring the languages validator). A scalar string is one tracker, not iterated
        character-by-character (`trackers: Nyaa` -> `{"nyaa"}`). A non-iterable scalar
        raises `ValueError` so it surfaces as a clean `ValidationError` instead of a
        raw `TypeError` that would escape the cli's error handler.
        """

        if isinstance(value, (str, list, set, tuple)) and not value:
            return PUBLIC_TRACKERS | PRIVATE_TRACKERS
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, (list, set, tuple)):
            raise ValueError("trackers must be a list of tracker names")
        return {str(tracker).casefold() for tracker in cast("list[Any]", value)}


# Languages applied to imported files. An explicit empty list in the config coalesces
# back to these (a file must never be imported with no language - Sonarr reads that as
# Unknown and may re-grab). Referenced by both the field default and the validator.
_LANGUAGES_DUAL_DEFAULT = ["Japanese", "English"]
_LANGUAGES_SINGLE_DEFAULT = ["Japanese"]


class ImportsSettings(_ConfigBase):
    """Wait-for-completion + manual import.

    The wait modes, the blocking monitor, and the manual import itself are Sonarr
    only. Pending-import tracking and `post_import_category` also cover Radarr
    grabs (which Radarr's own completed-download handling imports).
    """

    wait_mode: ImportWaitMode = ImportWaitMode.OFF
    """When, if ever, to wait for grabbed torrents to finish downloading and then drive Sonarr's import."""

    # ge=1: a zero timeout/poll cadence is a degenerate busy-loop, never a
    # documented disable (that's `wait_mode: off`).
    wait_timeout: int = Field(default=3600, ge=1)
    """Seconds to wait per torrent for qBittorrent to finish downloading it."""

    ready_timeout: int = Field(default=600, ge=1)
    """Seconds to then wait for Sonarr to rescan and import the finished files.

    Measured from the last file Pearlarr saw land rather than the download's
    completion, so a season pack imported file-by-file times out only after
    this long with no visible progress.
    """

    poll_interval: int = Field(default=30, ge=1)
    """Seconds between polls of qBittorrent and the Sonarr queue while waiting."""

    # The fast lane between heavy polls: re-reads /api/v3/episode + one batched
    # qBittorrent info read - NEVER the heavy RefreshMonitoredDownloads/queue.
    progress_poll_interval: int = Field(default=5, ge=0)
    """Seconds between lightweight wait-screen refreshes (progress bars, speed, ETA) within each poll interval.

    `0` disables the extra refreshes. Rows then advance once per
    `poll_interval` while spinners and timers still animate. A value at or
    above `poll_interval` behaves as disabled.
    """

    # Constrained so a typo is a clean ValidationError at load, not a Sonarr API error.
    mode: Literal["auto", "move", "copy"] = "auto"
    """How the import takes finished files into the library.

    `auto` lets Sonarr choose between moving and copying (a still-seeding
    torrent is copied). `move` and `copy` force it. Forwarded to Sonarr
    verbatim.
    """

    remove_from_queue: bool = True
    """Remove a torrent's leftover Sonarr queue entry once every record on it has imported.

    Sonarr clears an entry itself only when a single import covers the grab's
    full episode count, so a torrent finished across several passes (some
    files imported by Sonarr, the rest by Pearlarr) stays parked in its queue,
    where completed-download handling may import it again. Removing the entry
    closes that window: Sonarr records the download as manually ignored, and
    the torrent keeps seeding in qBittorrent. Sonarr only.
    """

    # Applied by the wait machinery, so it only fires when the run's resolved
    # wait mode is non-off.
    post_import_category: str | None = None
    """qBittorrent category applied once every import using the torrent has completed.

    Lets finished torrents carry their own seeding rules. Covers both arrs: a
    torrent shared by several entries, or by Sonarr and Radarr, moves only after
    all of them have imported. Point delete-with-data cleanup scripts at this
    category alone - a SeaDex update can attach a new entry to an already-moved
    torrent, and deleted data is re-downloaded. Blank keeps the add-time
    category. Ignored when `wait_mode` is `off`.
    """

    default_quality: str | None = None
    """Sonarr quality name, e.g. `Bluray-2160p`, filling the gaps when a file's quality cannot be detected.

    Useful on a 4K instance. Blank adds no default. A name matching no Sonarr
    quality definition is warned about and ignored.
    """

    languages_dual: list[str] = Field(default_factory=lambda: list(_LANGUAGES_DUAL_DEFAULT))
    """Languages tagged on files imported from dual-audio releases.

    An empty list falls back to the default. A file is never imported with no
    language (Sonarr would read it as Unknown and could re-grab it).
    """

    languages_single: list[str] = Field(default_factory=lambda: list(_LANGUAGES_SINGLE_DEFAULT))
    """Languages tagged on files imported from single-audio releases. Empty falls back like `languages_dual`."""

    # ge=1: 0 would expire every pending-import record immediately, silently
    # defeating the deferred-import carry.
    pending_max_age_days: int = Field(default=14, ge=1)
    """Days a pending import may wait for its download before its record is dropped."""

    # ge=1: 0 is degenerate (the wait view floors the cadence to the poll interval).
    digest_interval: int = Field(default=300, ge=1)
    """Target seconds between wait-progress digest lines on a non-interactive console (Docker, cron)."""

    @field_validator("wait_mode", mode="before")
    @classmethod
    def _yaml_bool_off(cls, value: Any) -> Any:
        """Map YAML's unquoted `off` back to the OFF mode.

        YAML 1.1 parses a bare `off` (the documented disabled value) as the boolean
        `False`, so without this `wait_mode: off` would fail enum validation and skip
        the whole run. `False` reads as "disabled", so it maps to OFF. Any other
        unrecognized value still raises cleanly.
        """

        if value is False:
            return ImportWaitMode.OFF
        return value

    @field_validator("languages_dual", "languages_single", mode="before")
    @classmethod
    def _languages_default_if_empty(cls, value: Any, info: ValidationInfo) -> Any:
        """Coalesce an explicit empty list to the language default.

        Preserves the pre-Pydantic truthiness coalescing (`value if value else
        default`): `languages_dual: []` in the config must not tag imported files with
        no language. A blank/absent key already takes the default_factory via the
        inherited blank-drop, so this only handles an explicitly-empty list.
        """

        if value:
            return value
        return list(
            _LANGUAGES_DUAL_DEFAULT if info.field_name == "languages_dual" else _LANGUAGES_SINGLE_DEFAULT,
        )


class NotificationsSettings(_ConfigBase):
    """Discord + generic webhook + the walk-away ping when a wait pass completes.

    Both webhook URLs are `SecretStr`: the URL itself embeds the
    credential (the Discord path IS the token), so a dumped/logged model masks
    them. The Notifier construction unwraps them.
    """

    discord_url: SecretStr | None = None
    """Discord webhook URL grab notifications are posted to. Blank disables them."""

    wait_webhook_url: SecretStr | None = None
    """Webhook URL (ntfy, gotify, Home Assistant, ...) receiving the wait-pass summary. Blank disables it."""

    wait_notify: bool = False
    """Send a notification when a wait pass completes.

    Absent, it turns on automatically whenever any webhook URL is set. An
    explicit value wins.
    """

    @model_validator(mode="before")
    @classmethod
    def _derive_wait_notify(cls, data: Any) -> Any:
        """Default `wait_notify` on whenever any webhook is set. An explicit value wins.

        Order-independent w.r.t. the inherited blank-drop: it keys off
        `wait_notify` being absent/None, true either way.
        """

        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, Any]", data)
        if raw.get("wait_notify") is None:
            raw = {
                **raw,
                "wait_notify": raw.get("discord_url") is not None or raw.get("wait_webhook_url") is not None,
            }
        return raw


class ScheduleSettings(_ConfigBase):
    """Scheduled-mode cadence (hours between cycles)."""

    interval_hours: float = Field(default=6.0, gt=0, allow_inf_nan=False)
    """Hours between scheduled cycles."""


type LogFormat = Literal["auto", "rich", "plain", "json"]
"""The console output formats `advanced.log_format` accepts.

`auto` resolves once at logger setup: rich on a TTY stdout, plain otherwise.
"""


class AdvancedSettings(_ConfigBase):
    """Advanced knobs (rate limiting, caching, run cap, logging)."""

    # ge=0: 0 disables the rate-limit sleep (load-bearing for the concurrent
    # episode fetch). A negative value would crash time.sleep mid-run.
    sleep_time: int = Field(default=0, ge=0)
    """Seconds slept between API queries, as rate limiting. `0` disables the sleep."""

    # ge=0: 0 = always re-download the mapping sources (a valid dev choice).
    cache_time: int = Field(default=1, ge=0)
    """Days downloaded mapping sources are kept before re-downloading. `0` re-downloads every run."""

    interactive: bool = False
    """Prompt for a choice when several torrents match, instead of taking the best automatically."""

    # ge=0: 0 disables the cap (the download-tool convention, per qBittorrent/Sonarr).
    max_torrents_to_add: int = Field(default=10, ge=0)
    """Cap on torrents added per run. `0` removes the cap.

    Keeps a first run on a large library from flooding qBittorrent. Later runs
    pick up where the cap stopped. Preview runs ignore the cap, so a preview
    always reports the whole library.
    """

    detect_arr_activity: bool = True
    """Re-check titles whose files Sonarr or Radarr changed since the last run.

    Polls each arr's history at run start. Turn off if a release you replaced
    arr-side keeps being re-grabbed.
    """

    # Constrained to the levels the logger honors, so a typo is a clean
    # ValidationError at load instead of a runtime warn-and-default.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Minimum level written to the console and the file log, case-insensitive."""

    log_format: LogFormat = "auto"
    """Console output format.

    `auto` resolves to `rich` on a terminal and `plain` when piped or under
    Docker. `plain` and `json` also disable the live progress views.
    """

    log_retention_days: int = Field(default=14, ge=1)
    """How many days of dated log backups to keep.

    Each run rotates the previous run's log to a dated backup. Backups older
    than this many days are deleted once the run's configuration loads.
    """

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: Any) -> Any:
        """Uppercase a configured level so `info` and `INFO` both validate."""

        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("log_format", mode="before")
    @classmethod
    def _lowercase_log_format(cls, value: Any) -> Any:
        """Lowercase a configured format so `JSON` and `json` both validate."""

        if isinstance(value, str):
            return value.lower()
        return value


class MappingsSettings(_ConfigBase):
    """ID/episode mapping sources: `False` disables, blank auto-downloads.

    `anime`/`anibridge` also accept an inline mapping dict (a power-user path the
    resolver supports). `anidb`'s inline form is an XML element that can't come from
    a YAML file, so only disable/auto-download are offered there.
    """

    anime_mappings: dict[str, Any] | Literal[False] | None = None
    """Kometa `Anime-IDs` source, a fallback for IDs the primary graph misses.

    Blank auto-downloads it. `false` disables it. An inline mapping is accepted.
    """

    anidb_mappings: Literal[False] | None = None
    """AniDB `anime-lists` XML source, a fallback used for specials. Blank auto-downloads it. `false` disables it."""

    anibridge_mappings: dict[str, Any] | Literal[False] | None = None
    """`anibridge-mappings` v3 graph, the primary source of IDs and per-season episode maps.

    Blank auto-downloads it. `false` disables it. An inline mapping is accepted.
    """


def _digest_canonical(value: object) -> Json:
    """A model-dump value as digest-stable JSON data.

    Sets are sorted (their iteration order is hash-seed dependent, so it differs
    between processes). Dict keys are left to `json.dumps(sort_keys=True)`. A
    `StrEnum` member passes through the `str` arm - `json.dumps` writes a str
    subclass's raw content, which IS the enum's plain value.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in cast("set[object] | frozenset[object]", value))
    if isinstance(value, (list, tuple)):
        return [_digest_canonical(item) for item in cast("list[object] | tuple[object, ...]", value)]
    if isinstance(value, dict):
        return {str(key): _digest_canonical(item) for key, item in cast("dict[object, object]", value).items()}
    return str(value)


class AppConfig(_ConfigBase):
    """The full validated config: one submodel per settings group.

    `config_version` is the one top-level scalar. `load` brings an older file's
    mapping forward (in memory) before validation, so removed or renamed keys
    from a previous schema never trip `extra="forbid"`.
    """

    config_version: int = Field(default=CONFIG_VERSION, ge=1)
    """Schema version of the config file's keys and values.

    Stamped by the starter template and `pearlarr config migrate`. A file from
    an older Pearlarr is migrated automatically in memory at every load.
    """

    sonarr: SonarrSettings = Field(default_factory=SonarrSettings)
    """Sonarr connection and behavior.

    `url` and `api_key` are required only when a Sonarr run actually executes.
    A Radarr-only config still loads.
    """

    radarr: ArrSettings = Field(default_factory=ArrSettings)
    """Radarr connection and behavior.

    `url` and `api_key` are required only when a Radarr run actually executes.
    A Sonarr-only config still loads.
    """

    qbittorrent: QbittorrentSettings = Field(default_factory=QbittorrentSettings)
    """qBittorrent connection.

    All of `host`, `username`, and `password` are needed to grab. Leave any
    blank to run in preview mode (everything is reported, nothing is added).
    """

    seadex: SeadexSettings = Field(default_factory=SeadexSettings)
    """SeaDex release-selection filters."""

    imports: ImportsSettings = Field(default_factory=ImportsSettings)
    """Waiting for grabbed torrents and Sonarr manual import. Sonarr only: a Radarr run ignores this group."""

    notifications: NotificationsSettings = Field(default_factory=NotificationsSettings)
    """Notification webhooks. Either, both, or neither may be set."""

    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    """Scheduled-mode cadence (the default command, `run scheduled`)."""

    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    """Rate limiting, caching, run caps, and logging."""

    mappings: MappingsSettings = Field(default_factory=MappingsSettings)
    """ID and episode mapping sources. `anibridge_mappings` is the primary. The others fill anything it misses."""

    # Set once by `load` after construction (frozen models still allow private-attr
    # assignment). The file checksum is the cache descriptor, the path feeds error text.
    _path: str = PrivateAttr(default="")
    _checksum: str = PrivateAttr(default="")
    _migration: MigrationOutcome | None = PrivateAttr(default=None)
    # Memoized on first `selection_digest()` call (immutable model). "" = not yet computed.
    _selection_digest: str = PrivateAttr(default="")

    @field_validator("config_version")
    @classmethod
    def _refuse_newer_schema(cls, value: int) -> int:
        """Refuse a file stamped by a newer Pearlarr, naming both versions."""

        if value > CONFIG_VERSION:
            raise ValueError(
                f"the file was written for a newer Pearlarr (schema version {value}; "
                f"this version reads up to {CONFIG_VERSION}) - upgrade Pearlarr",
            )
        return value

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        """Locate, load and validate the config file.

        Copies the bundled template to `path` and raises `FileNotFoundError` when
        the file is missing (so a first run writes a starter config), then parses,
        migrates an older file's mapping to the current schema (in memory only -
        see `migration`), overlays any `PEARLARR_*__*` environment overrides, and
        validates it. An invalid config raises `pydantic.ValidationError` (a
        `ValueError` subclass) naming the offending keys.
        """

        if not os.path.exists(path):
            write_starter_config(path)
            raise FileNotFoundError(f"No config file at {path}; a starter template was written - fill it in and re-run")

        with open(path, "rb") as f:
            raw = f.read()
        parsed, migration = _parse_and_migrate(raw)
        environ = os.environ
        overlay = _env_overlay(environ)
        # Env wins per leaf. A non-mapping file (junk) skips the merge so its own
        # validation error stands unchanged.
        if overlay and is_json_obj(parsed):
            _deep_merge(cast("dict[str, object]", parsed), overlay)
        config = cls.model_validate(parsed)
        # PrivateAttr writes on a frozen model: go through object.__setattr__ so the
        # type checkers don't read it as a frozen-field mutation (it isn't - these are
        # private attrs, set once here at load and never again).
        object.__setattr__(config, "_path", path)
        object.__setattr__(config, "_checksum", _config_checksum(raw, environ))
        object.__setattr__(config, "_migration", migration)
        return config

    def checksum(self) -> str:
        """MD5 hex digest of the config file's bytes at load time (cache descriptor)."""

        return self._checksum

    def selection_digest(self) -> str:
        """SHA-256 hex over the selection-affecting settings (the cache's re-check key).

        Covers the whole `seadex` group plus `imports.languages_*` - the settings
        that change which release a title should end up with. The cache compares
        this against the digest a previous full pass vouched for
        (`selection_stale`) and re-checks every cached verdict when it moved.
        Whole-group on purpose: a future seadex knob is covered by default.
        Over-inclusion costs one re-check run. Omission would leave stale verdicts.
        Memoized (the model is immutable) so the run's repeated reads hash once.
        """

        if not self._selection_digest:
            payload: dict[str, object] = {
                "seadex": self.seadex.model_dump(mode="python"),
                "languages_dual": self.imports.languages_dual,
                "languages_single": self.imports.languages_single,
            }
            canonical = json.dumps(_digest_canonical(payload), sort_keys=True, separators=(",", ":"))
            object.__setattr__(self, "_selection_digest", sha256(canonical.encode("utf-8")).hexdigest())
        return self._selection_digest

    def migration(self) -> MigrationOutcome | None:
        """The in-memory schema migration `load` applied, or None for a current file.

        Non-None means the file on disk still spells the old schema. The callers
        that report suggest `pearlarr config migrate` to rewrite it.
        """

        return self._migration

    def for_arr(self, arr: Arr) -> ArrSettings:
        """The per-arr connection/behavior submodel (one load serves both arrs)."""

        return self.sonarr if arr is Arr.SONARR else self.radarr

    def is_configured(self, arr: Arr) -> bool:
        """Whether the arr's connection pair (url + api_key) is filled in.

        The CLI uses this to skip an unconfigured arr cleanly (a Sonarr-only
        setup is normal) instead of tripping `require_connection`.
        """

        return not self.missing_arr_keys(arr)

    def missing_arr_keys(self, arr: Arr) -> tuple[str, ...]:
        """The unset halves of the arr's connection pair, as dotted config keys.

        Empty when the arr is fully configured. Exactly one key means a
        half-configured arr (almost certainly a mistake, which the CLI warns
        about by name rather than skipping silently).
        """

        sub = self.for_arr(arr)
        pairs = (("url", sub.url), ("api_key", sub.api_key))
        return tuple(f"{arr.value}.{field}" for field, value in pairs if not value)

    def require_connection(self, arr: Arr) -> tuple[str, str]:
        """The arr's `(url, api_key)`, or raise naming the missing key + file.

        Connection settings are optional at parse time so a config filled in for only
        one arr still validates. This enforces them lazily when that arr actually runs.
        """

        sub = self.for_arr(arr)
        if not sub.url:
            raise ValueError(f"{arr.value}.url must be set in {self._path}")
        if not sub.api_key:
            raise ValueError(f"{arr.value}.api_key must be set in {self._path}")
        return sub.url, sub.api_key.get_secret_value()


def _qualifying_env(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    """The config-override environment variables, sorted by name.

    A variable qualifies only when it starts with `PEARLARR_` and carries the
    `__` nesting delimiter after that prefix. A delimiter-less `PEARLARR_*` name
    (the data-dir and Docker operational vars) is reserved and never read as
    config. Sorted for a deterministic overlay and checksum.
    """

    return sorted(
        (name, value)
        for name, value in environ.items()
        if name.startswith(ENV_CONFIG_PREFIX) and ENV_CONFIG_DELIMITER in name.removeprefix(ENV_CONFIG_PREFIX)
    )


def _mapping_annotation(annotation: object) -> bool:
    """Whether the annotation is (or one union arm is) a free-form mapping."""

    if get_origin(annotation) is dict:
        return True
    if get_origin(annotation) in (UnionType, Union):
        return any(get_origin(arm) is dict for arm in get_args(annotation))
    return False


def _overlay_path(name: str) -> list[str]:
    """The key path a qualifying variable's name addresses.

    Model-addressed segments fold to lowercase (matching the file's key style).
    Once the path enters a free-form mapping field (e.g. `qbittorrent.options`,
    whose keys reach qbittorrentapi verbatim and case-sensitively), the
    remaining segments keep the variable's exact case.
    """

    model: type[BaseModel] | None = AppConfig
    freeform = False
    path: list[str] = []
    for segment in name.removeprefix(ENV_CONFIG_PREFIX).split(ENV_CONFIG_DELIMITER):
        if freeform:
            path.append(segment)
            continue
        lowered = segment.lower()
        path.append(lowered)
        field = model.model_fields.get(lowered) if model is not None else None
        annotation = field.annotation if field is not None else None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model = annotation
        else:
            model = None
            freeform = _mapping_annotation(annotation)
    return path


def _env_overlay(environ: Mapping[str, str]) -> dict[str, object]:
    """The environment's config-key overrides as a nested mapping.

    Each qualifying variable's name (minus the prefix) splits on `__` into a
    key path (see `_overlay_path`), and its value is parsed with `yaml.safe_load`
    so an env value means exactly what the same text means in `config.yml` (an
    empty string parses to None, which the blank-key handling then coalesces to
    the default). A malformed value raises a `ValidationError` naming only the
    variable, so it renders through the invalid-configuration arms like a bad
    file key.
    """

    overlay: dict[str, object] = {}
    for name, raw_value in _qualifying_env(environ):
        try:
            value: object = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            # Severed chain + no input attached: yaml's marks and the raw value
            # may both hold a secret.
            raise ValidationError.from_exception_data(
                "AppConfig",
                [
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "env_override", "is not valid YAML - fix or unset the environment variable"
                        ),
                        loc=(name,),
                        input=None,
                    )
                ],
                hide_input=True,
            ) from None
        path = _overlay_path(name)
        cursor = overlay
        for segment in path[:-1]:
            branch = cursor.get(segment)
            if not isinstance(branch, dict):
                fresh: dict[str, object] = {}
                cursor[segment] = fresh
                cursor = fresh
            else:
                cursor = cast("dict[str, object]", branch)
        cursor[path[-1]] = value
    return overlay


def _deep_merge(base: dict[str, object], overlay: Mapping[str, object]) -> None:
    """Merge `overlay` into `base` in place: a leaf overrides, two dicts merge per key.

    So `PEARLARR_SONARR__URL` replaces `sonarr.url` without clobbering the file's
    sibling `sonarr.api_key`.
    """

    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(cast("dict[str, object]", existing), cast("Mapping[str, object]", value))
        else:
            base[key] = value


def _config_checksum(raw: bytes, environ: Mapping[str, str]) -> str:
    """MD5 over the config bytes plus the qualifying env overrides (the cache descriptor).

    Folding the overrides in means an env change registers as a config change,
    so the cache re-checks against the effective settings. An unrelated env var
    leaves the digest untouched.
    """

    lines = "".join(f"{name}={value}\n" for name, value in _qualifying_env(environ))
    return md5(raw + lines.encode("utf-8"), usedforsecurity=False).hexdigest()


def _parse_and_migrate(raw: bytes) -> tuple[object, MigrationOutcome | None]:
    """Parse config bytes and bring an old-schema mapping forward, in place.

    The one parse pipeline `load` and `upgrade_config_file` share, so the run
    path and the file rewrite can never read the same bytes differently.
    Migration runs BEFORE validation: an older schema's removed keys/values
    would otherwise trip `extra="forbid"`. A non-mapping document passes
    through unmigrated for validation to reject.
    """

    parsed: object = yaml.safe_load(raw) or {}
    migration = migrate_mapping(parsed) if is_json_obj(parsed) else None
    return parsed, migration


class ConfigRewriteError(Exception):
    """`upgrade_config_file`'s render did not round-trip. Nothing was written."""


@dataclass(frozen=True)
class ConfigUpgrade:
    """What `upgrade_config_file` did: both fields are None for an already-current file."""

    migration: MigrationOutcome | None
    backup_path: str | None


def _write_owner_only(path: str, data: bytes) -> None:
    """Write a secret-bearing file created 0600, never briefly looser.

    `O_CREAT` with mode 0600 closes the umask window an open-then-chmod
    creation would leave. The follow-up chmod covers the pre-existing-file
    case, where `O_TRUNC` keeps the old mode.
    """

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    restrict_config_permissions(path)


def upgrade_config_file(path: str) -> ConfigUpgrade:
    """Rewrite an older config file at the current schema, keeping a backup.

    The rewrite is the current annotated template with this file's values (and
    the schema fixes a load applies in memory) spliced in. The previous bytes
    land beside it first, as `<path>.bak`. Both new files are created
    owner-only (they carry API keys) and the swap goes through a temp file +
    `os.replace`, so a failed write can never leave a truncated config. A file
    already at the current version is not touched at all.

    Nothing is written unless the rendered text parses back to the exact
    settings that were validated - a splice defect refuses cleanly
    (`ConfigRewriteError`) instead of corrupting the file. Raises like
    `AppConfig.load` (`OSError` / `yaml.YAMLError` / `ValidationError`): a
    file this version cannot fully read is never rewritten.
    """

    with open(path, "rb") as f:
        raw = f.read()
    parsed, migration = _parse_and_migrate(raw)
    validated = AppConfig.model_validate(parsed)
    # The narrowing re-check is for the types: a non-mapping document cannot
    # have validated above.
    if migration is None or not is_json_obj(parsed):
        return ConfigUpgrade(migration=None, backup_path=None)

    rendered = render_migrated_config(starter_template_text(), parsed)
    try:
        reparsed: object = yaml.safe_load(rendered)
        drifted = AppConfig.model_validate(reparsed).model_dump() != validated.model_dump()
    except (yaml.YAMLError, ValidationError):
        drifted = True
    if drifted:
        raise ConfigRewriteError(
            f"Rewriting {path} would have changed its settings - left untouched "
            f"(this is a Pearlarr bug; please report it)",
        )

    backup = path + ".bak"
    _write_owner_only(backup, raw)
    tmp = path + ".tmp"
    try:
        _write_owner_only(tmp, rendered.encode("utf-8"))
        os.replace(tmp, path)
    except OSError:
        # Never leave a torn, secret-bearing temp file behind. The config
        # itself is untouched and the backup is a faithful copy of it.
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return ConfigUpgrade(migration=migration, backup_path=backup)
