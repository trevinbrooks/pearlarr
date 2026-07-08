"""Application configuration: typed, validated settings loaded from the YAML file.

``AppConfig`` is a Pydantic model tree over the config file. Pydantic owns parsing,
defaults, coercion (including the present-but-blank-YAML-key footgun) and validation:
an unknown/typo'd key or a bad value raises a clear ``ValidationError`` instead of
being silently dropped or coalesced. The rest of the package reads
``config.seadex.want_best`` rather than ``config.get("want_best", True)``.

Each settings group is its own submodel (``sonarr``/``radarr``/``qbittorrent``/
``seadex``/``imports``/``notifications``/``advanced``/``mappings``). The connection
``url``/``api_key`` for each arr are optional at parse time and enforced lazily at
point-of-use via :meth:`AppConfig.require_connection`, so a config filled in for only
one arr still validates and runs.
"""

import contextlib
import os
import shutil
from enum import StrEnum
from hashlib import md5
from typing import Any, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .manual_import import ImportWaitMode

# Tracker name classification. Stored casefolded so membership tests match the
# casefolded ``seadex.trackers`` setting and the casefolded tracker names from SeaDex.
PUBLIC_TRACKERS = {
    tracker.casefold()
    for tracker in [
        "Nyaa",
        "AnimeTosho",
        "AniDex",
        "RuTracker",
    ]
}

PRIVATE_TRACKERS = {
    tracker.casefold()
    for tracker in [
        "AB",
        "BeyondHD",
        "PassThePopcorn",
        "BroadcastTheNet",
        "HDBits",
        "Blutopia",
        "Aither",
    ]
}

# Every name the seadex ``Tracker`` enum can emit (casefolded), so the cli can warn on
# configured trackers that match nothing. Literals keep this module's imports light;
# a parity test pins them against the installed enum.
KNOWN_TRACKERS = PUBLIC_TRACKERS | PRIVATE_TRACKERS | {"other", "otherprivate"}

CONFIG_TEMPLATE_FILE = "config_sample.yml"


class Arr(StrEnum):
    """Which *arr the run targets.

    A ``StrEnum`` so the value still equals its string (``Arr.SONARR == "sonarr"``),
    serializes as a bare JSON cache key, and selects the matching config group -
    while making the two valid arrs the only representable states.
    """

    SONARR = "sonarr"
    RADARR = "radarr"


class PrivateReleaseAction(StrEnum):
    """The ``seadex.private_releases`` policy for private-tracker releases.

    Private releases are never grabbed: SeaDex carries no downloadable link or
    infohash for them, and no private-tracker auth is supported. WARN (default)
    warns and skips without caching when the preferred release is private-only,
    so the title is re-checked every run until a public release appears.
    FALLBACK grabs the entry's best public alternative instead, warning only
    when none exists at all. Titles satisfied by a fallback are remembered (a
    cache marker); switching back to WARN re-checks them and resurfaces the
    private-only warning.
    """

    WARN = "warn"
    FALLBACK = "fallback"


def template_path() -> str:
    """Absolute path to the bundled config template shipped beside this module."""

    return os.path.join(os.path.dirname(__file__), CONFIG_TEMPLATE_FILE)


def restrict_config_permissions(path: str) -> None:
    """Best-effort ``chmod 0600``: the config carries plaintext API keys.

    Both creation paths (the first-run template copy and ``config init``) call
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
    """Unwrap an optional :class:`SecretStr` at a point of use."""

    return secret.get_secret_value() if secret is not None else None


class _ConfigBase(BaseModel):
    """Shared base for every settings group: strict, immutable, blank-tolerant.

    ``extra="forbid"`` turns a typo'd key into a ``ValidationError`` (the whole point
    of validating); ``frozen=True`` matches the project's value-object convention and
    makes sharing one loaded config across both arrs provably safe;
    ``hide_input_in_errors=True`` keeps a rejected value (which could be a
    credential pasted under the wrong key) out of the ValidationError text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_none(cls, data: Any) -> Any:
        """Let blank YAML keys fall back to their declared default.

        A present-but-blank YAML key (``foo:``) parses to ``None``; a plain field
        default would not apply (the key is present) and ``None`` would be rejected
        by an ``int``/``str`` field. Drop ``None`` *only for known fields* so the
        default applies. Unknown keys are kept regardless of blankness, so
        ``extra="forbid"`` still flags both ``typo: value`` and a bare ``typo:``.
        """

        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, Any]", data)
        fields = cls.model_fields
        return {key: value for key, value in raw.items() if value is not None or key not in fields}


class ArrSettings(_ConfigBase):
    """Connection + per-arr behaviour, shared by the ``sonarr`` and ``radarr`` groups.

    ``api_key`` is a :class:`SecretStr` so a dumped/logged model masks it;
    :meth:`AppConfig.require_connection` (and the Sonarr->Radarr cross-check)
    unwrap it at the client-construction boundary. ``verify_ssl: false`` is the
    last-resort escape hatch for a self-signed arr whose CA can't be trusted via
    the OS store / ``SSL_CERT_FILE``.
    """

    url: str | None = None
    api_key: SecretStr | None = None
    verify_ssl: bool = True
    ignore_unmonitored: bool = False
    torrent_category: str | None = None


class SonarrSettings(ArrSettings):
    """Sonarr adds one cross-arr flag the Radarr group must not accept.

    ``ignore_movies_in_radarr`` is read only on a Sonarr run; declaring it here (not
    on the shared base) means ``extra="forbid"`` correctly rejects it under ``radarr``.
    """

    ignore_movies_in_radarr: bool = False


class QbittorrentSettings(_ConfigBase):
    """qBittorrent connection. The connection fields are modelled explicitly; ``options``
    is a scoped escape hatch for the remaining ``qbittorrentapi.Client`` kwargs (e.g.
    ``VERIFY_WEBUI_CERTIFICATE`` for a self-signed-HTTPS WebUI, ``REQUESTS_ARGS``) so the
    explicit model doesn't drop connectivity the old free-form ``qbit_info`` splat allowed.
    """

    host: str | None = None
    username: str | None = None
    password: SecretStr | None = None
    tags: list[str] | None = None
    # Extra keyword arguments splatted into qbittorrentapi.Client alongside the
    # connection triple. Empty by default; nest advanced client kwargs here.
    options: dict[str, Any] = Field(default_factory=dict)

    def credentials(self) -> tuple[str, str, str] | None:
        """The ``(host, username, password)`` triple, or ``None`` if any part is unset.

        All three are needed to add torrents; a missing one means run in preview mode
        (nothing is grabbed). The caller builds the client with explicit kwargs; the
        password is unwrapped here, at its point of use.
        """

        if self.host and self.username and self.password:
            return self.host, self.username, self.password.get_secret_value()
        return None


class SeadexSettings(_ConfigBase):
    """SeaDex release-selection filters."""

    private_releases: PrivateReleaseAction = PrivateReleaseAction.WARN
    prefer_dual_audio: bool = True
    want_best: bool = True
    ignore_tags: list[str] = Field(default_factory=list)
    # Default to all trackers (public + private) when none configured. Private are
    # included even when private releases won't be grabbed: they're filtered later,
    # after the overlap check against what's already downloaded.
    trackers: set[str] = Field(default_factory=lambda: PUBLIC_TRACKERS | PRIVATE_TRACKERS)
    ignore_anilist_ids: set[int] = Field(default_factory=set[int])
    ignore_seadex_update_times: bool = False
    use_torrent_hash_to_filter: bool = False

    @field_validator("trackers", mode="before")
    @classmethod
    def _casefold_trackers(cls, value: Any) -> Any:
        """Casefold explicit trackers so membership tests match SeaDex's names.

        An absent ``trackers`` key takes the ``PUBLIC | PRIVATE`` default_factory; an
        explicit empty list/string coalesces to that same default here (empty means "no
        restriction", never "match nothing" - which would silently grab nothing,
        mirroring the languages validator). A scalar string is one tracker, not iterated
        character-by-character (``trackers: Nyaa`` -> ``{"nyaa"}``); a non-iterable scalar
        raises ``ValueError`` so it surfaces as a clean ``ValidationError`` instead of a
        raw ``TypeError`` that would escape the cli's error handler.
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
# Unknown and may re-grab); referenced by both the field default and the validator.
_LANGUAGES_DUAL_DEFAULT = ["Japanese", "English"]
_LANGUAGES_SINGLE_DEFAULT = ["Japanese"]


class ImportsSettings(_ConfigBase):
    """Wait-for-completion + Sonarr manual import (Sonarr only; Radarr is a no-op)."""

    # off (disabled, default) / deferred / blocking / hybrid. An unrecognized value
    # raises ValidationError like any other bad config (surfaced cleanly by the cli).
    wait_mode: ImportWaitMode = ImportWaitMode.OFF
    # ge=1: a zero timeout/poll cadence is a degenerate busy-loop, never a
    # documented disable (that's ``wait_mode: off``).
    wait_timeout: int = Field(default=3600, ge=1)
    ready_timeout: int = Field(default=600, ge=1)
    poll_interval: int = Field(default=30, ge=1)
    # Fast-lane cockpit refresh cadence (seconds), between heavy polls: re-reads
    # /api/v3/episode for in-flight imports ("files inserted" bar) and does one
    # batched qBittorrent info read for in-flight downloads (bar/speed/ETA) -
    # NEVER the heavy RefreshMonitoredDownloads/queue. 0 disables it: rows then
    # advance once per ``poll_interval`` (spinner/timer still animate). A value
    # >= ``poll_interval`` is the same as disabled.
    progress_poll_interval: int = Field(default=5, ge=0)
    # Sonarr importMode, forwarded verbatim (sonarr_client.manual_import_execute).
    # Constrained so a typo is a clean ValidationError at load, not a Sonarr API error.
    mode: Literal["auto", "move", "copy"] = "auto"
    # qBittorrent category applied to a torrent once its import is verified
    # complete (e.g. to hand it to different seeding rules). Blank leaves the
    # add-time category in place. Applied by the wait machinery, so it only
    # fires when the run's resolved wait mode is non-off.
    post_import_category: str | None = None
    default_quality: str | None = None
    languages_dual: list[str] = Field(default_factory=lambda: list(_LANGUAGES_DUAL_DEFAULT))
    languages_single: list[str] = Field(default_factory=lambda: list(_LANGUAGES_SINGLE_DEFAULT))
    # ge=1: 0 would expire every pending-import record immediately, silently
    # defeating the deferred-import carry.
    pending_max_age_days: int = Field(default=14, ge=1)
    # ge=1: 0 is degenerate (the wait view floors the cadence to the poll interval).
    digest_interval: int = Field(default=300, ge=1)

    @field_validator("wait_mode", mode="before")
    @classmethod
    def _yaml_bool_off(cls, value: Any) -> Any:
        """Map YAML's unquoted ``off`` back to the OFF mode.

        YAML 1.1 parses a bare ``off`` (the documented disabled value) as the boolean
        ``False``, so without this ``wait_mode: off`` would fail enum validation and skip
        the whole run. ``False`` reads as "disabled", so it maps to OFF; any other
        unrecognized value still raises cleanly.
        """

        if value is False:
            return ImportWaitMode.OFF
        return value

    @field_validator("languages_dual", "languages_single", mode="before")
    @classmethod
    def _languages_default_if_empty(cls, value: Any, info: ValidationInfo) -> Any:
        """Coalesce an explicit empty list to the language default.

        Preserves the pre-Pydantic truthiness coalescing (``value if value else
        default``): ``languages_dual: []`` in the config must not tag imported files with
        no language. A blank/absent key already takes the default_factory via the
        inherited blank-drop, so this only handles an explicitly-empty list.
        """

        if value:
            return value
        return list(
            _LANGUAGES_DUAL_DEFAULT if info.field_name == "languages_dual" else _LANGUAGES_SINGLE_DEFAULT,
        )


class NotificationsSettings(_ConfigBase):
    """Discord + generic webhook + the walk-away wait-complete ping.

    Both webhook URLs are :class:`SecretStr`: the URL itself embeds the
    credential (the Discord path IS the token), so a dumped/logged model masks
    them; the Notifier construction unwraps them.
    """

    discord_url: SecretStr | None = None
    wait_webhook_url: SecretStr | None = None
    wait_notify: bool = False

    @model_validator(mode="before")
    @classmethod
    def _derive_wait_notify(cls, data: Any) -> Any:
        """Default ``wait_notify`` on whenever any webhook is set; explicit value wins.

        Order-independent w.r.t. the inherited blank-drop: it keys off
        ``wait_notify`` being absent/None, true either way.
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


type LogFormat = Literal["auto", "rich", "plain", "json"]
"""The console output formats ``advanced.log_format`` accepts.

``auto`` resolves once at logger setup: rich on a TTY stdout, plain otherwise.
"""


class AdvancedSettings(_ConfigBase):
    """Advanced knobs (rate limiting, caching, run cap, logging)."""

    # ge=0: 0 disables the rate-limit sleep (load-bearing for the concurrent
    # episode fetch); a negative value would crash time.sleep mid-run.
    sleep_time: int = Field(default=2, ge=0)
    # ge=0: 0 = always re-download the mapping sources (a valid dev choice).
    cache_time: int = Field(default=1, ge=0)
    interactive: bool = False
    # ge=1 when set: a cap of 0 would silently grab nothing (None = unlimited).
    max_torrents_to_add: int | None = Field(default=None, ge=1)
    # Poll each arr's history at run start and re-check entries whose files the
    # arr changed since the last pass. Opt-out: the re-check can re-grab a
    # release the user deliberately replaced arr-side.
    detect_arr_activity: bool = True
    # Constrained to the levels the logger honors, so a typo is a clean
    # ValidationError at load instead of a runtime warn-and-default.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: LogFormat = "auto"

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: Any) -> Any:
        """Uppercase a configured level so ``info`` and ``INFO`` both validate."""

        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("log_format", mode="before")
    @classmethod
    def _lowercase_log_format(cls, value: Any) -> Any:
        """Lowercase a configured format so ``JSON`` and ``json`` both validate."""

        if isinstance(value, str):
            return value.lower()
        return value


class MappingsSettings(_ConfigBase):
    """ID/episode mapping sources: ``False`` disables, blank auto-downloads.

    ``anime``/``anibridge`` also accept an inline mapping dict (a power-user path the
    resolver supports). ``anidb``'s inline form is an XML element that can't come from
    a YAML file, so only disable/auto-download are offered there.
    """

    anime_mappings: dict[str, Any] | Literal[False] | None = None
    anidb_mappings: Literal[False] | None = None
    anibridge_mappings: dict[str, Any] | Literal[False] | None = None


class AppConfig(_ConfigBase):
    """The full validated config: one submodel per settings group."""

    sonarr: SonarrSettings = Field(default_factory=SonarrSettings)
    radarr: ArrSettings = Field(default_factory=ArrSettings)
    qbittorrent: QbittorrentSettings = Field(default_factory=QbittorrentSettings)
    seadex: SeadexSettings = Field(default_factory=SeadexSettings)
    imports: ImportsSettings = Field(default_factory=ImportsSettings)
    notifications: NotificationsSettings = Field(default_factory=NotificationsSettings)
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    mappings: MappingsSettings = Field(default_factory=MappingsSettings)

    # Set once by ``load`` after construction (frozen models still allow private-attr
    # assignment); the file checksum is the cache descriptor, the path feeds error text.
    _path: str = PrivateAttr(default="")
    _checksum: str = PrivateAttr(default="")

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        """Locate, load and validate the config file.

        Copies the bundled template to ``path`` and raises ``FileNotFoundError`` when
        the file is missing (so a first run writes a starter config), then parses and
        validates it. An invalid config raises ``pydantic.ValidationError`` (a
        ``ValueError`` subclass) naming the offending keys.

        Args:
            path (str): Path to the config file.
        """

        if not os.path.exists(path):
            shutil.copy(template_path(), path)
            restrict_config_permissions(path)
            raise FileNotFoundError(f"No config file at {path}; a starter template was written - fill it in and re-run")

        with open(path, "rb") as f:
            raw = f.read()
        config = cls.model_validate(yaml.safe_load(raw) or {})
        # PrivateAttr writes on a frozen model: go through object.__setattr__ so the
        # type checkers don't read it as a frozen-field mutation (it isn't - these are
        # private attrs, set once here at load and never again).
        object.__setattr__(config, "_path", path)
        object.__setattr__(config, "_checksum", md5(raw, usedforsecurity=False).hexdigest())
        return config

    def checksum(self) -> str:
        """MD5 hex digest of the config file's bytes at load time (cache descriptor)."""

        return self._checksum

    def for_arr(self, arr: Arr) -> ArrSettings:
        """The per-arr connection/behaviour submodel (one load serves both arrs)."""

        return self.sonarr if arr is Arr.SONARR else self.radarr

    def is_configured(self, arr: Arr) -> bool:
        """Whether the arr's connection pair (url + api_key) is filled in.

        The CLI uses this to skip an unconfigured arr cleanly (a Sonarr-only
        setup is normal) instead of tripping :meth:`require_connection`.
        """

        return not self.missing_arr_keys(arr)

    def missing_arr_keys(self, arr: Arr) -> tuple[str, ...]:
        """The unset halves of the arr's connection pair, as dotted config keys.

        Empty when the arr is fully configured; exactly one key means a
        half-configured arr (almost certainly a mistake, which the CLI warns
        about by name rather than skipping silently).
        """

        sub = self.for_arr(arr)
        pairs = (("url", sub.url), ("api_key", sub.api_key))
        return tuple(f"{arr.value}.{field}" for field, value in pairs if not value)

    def require_connection(self, arr: Arr) -> tuple[str, str]:
        """The arr's ``(url, api_key)``, or raise naming the missing key + file.

        Connection settings are optional at parse time so a config filled in for only
        one arr still validates; this enforces them lazily when that arr actually runs.
        """

        sub = self.for_arr(arr)
        if not sub.url:
            raise ValueError(f"{arr.value}.url must be set in {self._path}")
        if not sub.api_key:
            raise ValueError(f"{arr.value}.api_key must be set in {self._path}")
        return sub.url, sub.api_key.get_secret_value()
