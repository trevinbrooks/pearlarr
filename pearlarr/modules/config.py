"""Application configuration: typed, validated settings loaded from the YAML file.

``AppConfig`` is a Pydantic model tree over the config file. Pydantic owns parsing,
defaults, coercion (including the present-but-blank-YAML-key footgun) and validation:
an unknown/typo'd key or a bad value raises a clear ``ValidationError`` instead of
being silently dropped or coalesced. The rest of the package reads
``config.seadex.want_best`` rather than ``config.get("want_best", True)``.

Each settings group is its own submodel (``sonarr``/``radarr``/``qbittorrent``/
``seadex``/``imports``/``notifications``/``advanced``/``mappings``). The connection
``url``/``api_key`` for each arr are optional at parse time and enforced lazily at
point-of-use via ``AppConfig.require_connection``, so a config filled in for only
one arr still validates and runs.
"""

import contextlib
import os
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

# Tracker name classification. The *_TRACKER_NAMES tuples keep SeaDex's exact casings
# (the docs generator renders them into the sample and reference); the sets are
# casefolded so membership tests match the casefolded ``seadex.trackers`` setting and
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

# Every name the seadex ``Tracker`` enum can emit (casefolded), so the cli can warn on
# configured trackers that match nothing. Literals keep this module's imports light;
# a parity test pins them against the installed enum.
KNOWN_TRACKERS = PUBLIC_TRACKERS | PRIVATE_TRACKERS | {tracker.casefold() for tracker in OTHER_TRACKER_NAMES}

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
    infohash for them, and no private-tracker auth is supported. The policy
    decides what happens when a title's preferred release is private-only.
    """

    WARN = "warn"
    """Warn and skip without caching, so the title is re-checked every run until a public release appears."""

    FALLBACK = "fallback"
    """Grab the entry's best public alternative instead; warn only when none exists at all."""


def template_path() -> str:
    """Absolute path to the bundled config template shipped beside this module."""

    return os.path.join(os.path.dirname(__file__), CONFIG_TEMPLATE_FILE)


def write_starter_config(path: str) -> None:
    """Copy the bundled template to ``path`` as a user-owned starter config.

    Drops the template's generated-file banner (the copy is the user's file to
    edit, not a generated artifact) and restricts the copy to owner-only.
    """

    with open(template_path(), encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("# GENERATED")]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    restrict_config_permissions(path)


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
    """Unwrap an optional ``SecretStr`` at a point of use."""

    return secret.get_secret_value() if secret is not None else None


def strip_userinfo(value: str) -> str:
    """Mask a ``user:pass@`` login embedded in a URL/host value.

    Every user-facing rendering of a configured URL (``config show``, arr
    connection errors) goes through this: the login is a credential, the
    host is what the user needs to see.
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

    ``extra="forbid"`` turns a typo'd key into a ``ValidationError`` (the whole point
    of validating); ``frozen=True`` matches the project's value-object convention and
    makes sharing one loaded config across both arrs provably safe;
    ``hide_input_in_errors=True`` keeps a rejected value (which could be a
    credential pasted under the wrong key) out of the ValidationError text;
    ``use_attribute_docstrings=True`` lifts each field's attribute docstring into
    ``FieldInfo.description`` - the one authored home the docs generator renders.
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

    ``api_key`` is a ``SecretStr`` so a dumped/logged model masks it;
    ``AppConfig.require_connection`` (and the Sonarr->Radarr cross-check)
    unwrap it at the client-construction boundary. ``verify_ssl: false`` is the
    last-resort escape hatch for a self-signed arr whose CA can't be trusted via
    the OS store / ``SSL_CERT_FILE``.
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

    For Radarr this means unmonitored movies; for Sonarr, unmonitored series or
    entries whose episodes are all unmonitored.
    """

    torrent_category: str | None = None
    """qBittorrent category applied to torrents grabbed for this arr. Blank applies no category."""


class SonarrSettings(ArrSettings):
    """Sonarr adds one cross-arr flag the Radarr group must not accept.

    ``ignore_movies_in_radarr`` is read only on a Sonarr run; declaring it here (not
    on the shared base) means ``extra="forbid"`` correctly rejects it under ``radarr``.
    """

    ignore_movies_in_radarr: bool = False
    """Skip movies that already exist in Radarr instead of also grabbing them on the Sonarr side."""


class QbittorrentSettings(_ConfigBase):
    """qBittorrent connection. The connection fields are modelled explicitly; ``options``
    is a scoped escape hatch for the remaining ``qbittorrentapi.Client`` kwargs (e.g.
    ``VERIFY_WEBUI_CERTIFICATE`` for a self-signed-HTTPS WebUI, ``REQUESTS_ARGS``) so the
    explicit model doesn't drop connectivity the old free-form ``qbit_info`` splat allowed.
    """

    host: str | None = None
    """WebUI address of qBittorrent, e.g. `http://localhost:8080`.

    Leave any of `host`, `username` or `password` blank to run in preview mode:
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
    """What to do when a title's preferred release sits only on private trackers.

    Private releases are never grabbed: SeaDex carries no download link for
    them, and no private-tracker auth is supported.
    """

    prefer_dual_audio: bool = True
    """Prefer dual-audio releases; when off, prefer Japanese-audio releases."""

    want_best: bool = True
    """Prefer releases SeaDex marks as best."""

    ignore_tags: list[str] = Field(default_factory=list)
    """SeaDex release tags to skip, e.g. `Dolby Vision` or `Deband Required`. Empty skips none."""

    # Private trackers are in the default even though private releases won't be
    # grabbed: they're filtered later, after the overlap check against what's
    # already downloaded.
    trackers: set[str] = Field(default_factory=lambda: PUBLIC_TRACKERS | PRIVATE_TRACKERS)
    """Tracker names considered during release selection, case-insensitive.

    Empty or absent considers every supported tracker. A listed tracker is not
    necessarily grabbable: private releases are never downloaded, and grabs come
    only from the public trackers Pearlarr can parse.
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

    wait_mode: ImportWaitMode = ImportWaitMode.OFF
    """When, if ever, to wait for grabbed torrents to finish downloading and then drive Sonarr's import."""

    # ge=1: a zero timeout/poll cadence is a degenerate busy-loop, never a
    # documented disable (that's ``wait_mode: off``).
    wait_timeout: int = Field(default=3600, ge=1)
    """Seconds to wait per torrent for qBittorrent to finish downloading it."""

    ready_timeout: int = Field(default=600, ge=1)
    """Seconds to then wait for Sonarr to rescan and import the finished files."""

    poll_interval: int = Field(default=30, ge=1)
    """Seconds between polls of qBittorrent and the Sonarr queue while waiting."""

    # The fast lane between heavy polls: re-reads /api/v3/episode + one batched
    # qBittorrent info read - NEVER the heavy RefreshMonitoredDownloads/queue.
    progress_poll_interval: int = Field(default=5, ge=0)
    """Seconds between lightweight wait-screen refreshes (progress bars, speed, ETA) within each poll interval.

    `0` disables the extra refreshes; rows then advance once per
    `poll_interval` while spinners and timers still animate. A value at or
    above `poll_interval` behaves as disabled.
    """

    # Constrained so a typo is a clean ValidationError at load, not a Sonarr API error.
    mode: Literal["auto", "move", "copy"] = "auto"
    """How the import takes finished files into the library.

    `auto` lets Sonarr choose between moving and copying (a still-seeding
    torrent is copied); `move` and `copy` force it. Forwarded to Sonarr
    verbatim.
    """

    # Applied by the wait machinery, so it only fires when the run's resolved
    # wait mode is non-off.
    post_import_category: str | None = None
    """qBittorrent category applied to a torrent once its import is verified complete.

    Useful to give finished torrents different seeding rules. Blank keeps
    the add-time category. Only applies when `wait_mode` is not `off`.
    """

    default_quality: str | None = None
    """Sonarr quality name, e.g. `Bluray-2160p`, filling the gaps when a file's quality cannot be detected.

    Useful on a 4K instance. Blank adds no default; a name matching no Sonarr
    quality definition is warned about and ignored.
    """

    languages_dual: list[str] = Field(default_factory=lambda: list(_LANGUAGES_DUAL_DEFAULT))
    """Languages tagged on files imported from dual-audio releases.

    An empty list falls back to the default; a file is never imported with no
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
    """Discord + generic webhook + the walk-away ping when a wait pass completes.

    Both webhook URLs are ``SecretStr``: the URL itself embeds the
    credential (the Discord path IS the token), so a dumped/logged model masks
    them; the Notifier construction unwraps them.
    """

    discord_url: SecretStr | None = None
    """Discord webhook URL grab notifications are posted to. Blank disables them."""

    wait_webhook_url: SecretStr | None = None
    """Webhook URL (ntfy, gotify, Home Assistant, ...) receiving the wait-pass summary. Blank disables it."""

    wait_notify: bool = False
    """Send a notification when a wait pass completes.

    Absent, it turns on automatically whenever any webhook URL is set; an
    explicit value wins.
    """

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
    """Hours between scheduled cycles."""


type LogFormat = Literal["auto", "rich", "plain", "json"]
"""The console output formats ``advanced.log_format`` accepts.

``auto`` resolves once at logger setup: rich on a TTY stdout, plain otherwise.
"""


class AdvancedSettings(_ConfigBase):
    """Advanced knobs (rate limiting, caching, run cap, logging)."""

    # ge=0: 0 disables the rate-limit sleep (load-bearing for the concurrent
    # episode fetch); a negative value would crash time.sleep mid-run.
    sleep_time: int = Field(default=2, ge=0)
    """Seconds slept between API queries, as rate limiting. `0` disables the sleep."""

    # ge=0: 0 = always re-download the mapping sources (a valid dev choice).
    cache_time: int = Field(default=1, ge=0)
    """Days downloaded mapping sources are kept before re-downloading. `0` re-downloads every run."""

    interactive: bool = False
    """Prompt for a choice when several torrents match, instead of taking the best automatically."""

    # ge=1 when set: a cap of 0 would silently grab nothing (None = unlimited).
    max_torrents_to_add: int | None = Field(default=None, ge=1)
    """Cap on torrents added per run. Blank means unlimited."""

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
    Docker; `plain` and `json` also disable the live progress views.
    """

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
    """Kometa `Anime-IDs` source, a fallback for IDs the primary graph misses.

    Blank auto-downloads it; `false` disables it; an inline mapping is accepted.
    """

    anidb_mappings: Literal[False] | None = None
    """AniDB `anime-lists` XML source, a fallback used for specials. Blank auto-downloads it; `false` disables it."""

    anibridge_mappings: dict[str, Any] | Literal[False] | None = None
    """`anibridge-mappings` v3 graph, the primary source of IDs and per-season episode maps.

    Blank auto-downloads it; `false` disables it; an inline mapping is accepted.
    """


class AppConfig(_ConfigBase):
    """The full validated config: one submodel per settings group."""

    sonarr: SonarrSettings = Field(default_factory=SonarrSettings)
    """Sonarr connection and behaviour.

    `url` and `api_key` are required only when a Sonarr run actually executes;
    a Radarr-only config still loads.
    """

    radarr: ArrSettings = Field(default_factory=ArrSettings)
    """Radarr connection and behaviour.

    `url` and `api_key` are required only when a Radarr run actually executes;
    a Sonarr-only config still loads.
    """

    qbittorrent: QbittorrentSettings = Field(default_factory=QbittorrentSettings)
    """qBittorrent connection.

    All of `host`, `username` and `password` are needed to grab; leave any
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
    """ID and episode mapping sources. `anibridge_mappings` is the primary; the others fill anything it misses."""

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
            write_starter_config(path)
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
