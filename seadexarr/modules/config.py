"""Application configuration: file lifecycle and typed settings.

``AppConfig`` owns everything to do with the YAML config *file* — copying the
bundled template when it's missing, loading it, and keeping its key order in
sync with the template — and exposes the individual settings as typed, normalized
properties so the rest of the package reads ``config.public_only`` instead of
``config.get("public_only", True)`` scattered across the codebase.

Extracted from ``SeaDexArr.__init__`` / ``verify_config`` / ``setup_cache``
during the refactor; behaviour-preserving.
"""

import copy
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from hashlib import md5
from typing import Any
from xml.etree import ElementTree

import yaml
from ruamel.yaml import YAML

from .anibridge import AniBridgeGraph
from .manual_import import ImportWaitMode
from .mappings import AnimeIdsMap

# Tracker name classification. Stored casefolded so membership tests match the
# casefolded ``trackers`` setting and the casefolded tracker names from SeaDex.
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

CONFIG_TEMPLATE_FILE = "config_sample.yml"


class Arr(StrEnum):
    """Which *arr the run targets.

    A ``StrEnum`` so the value still equals its string (``Arr.SONARR ==
    "sonarr"``), serializes as a bare JSON cache key, and builds the
    ``{arr}_``-prefixed config keys unchanged - while making the two valid
    arrs the only representable states (no runtime ``ALLOWED_ARRS`` guard
    needed; an out-of-domain value is now a static type error).
    """

    SONARR = "sonarr"
    RADARR = "radarr"


def _template_path() -> str:
    """Absolute path to the bundled config template shipped beside this module."""

    return os.path.join(os.path.dirname(__file__), CONFIG_TEMPLATE_FILE)


@dataclass
class AppConfig:
    """Typed view over the loaded config file.

    ``data`` is the raw parsed mapping (kept available for the few arr-specific
    keys read directly by the Sonarr/Radarr subclasses); the properties expose
    the shared settings with their defaults and normalization applied.
    """

    path: str
    arr: Arr
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str, arr: Arr) -> "AppConfig":
        """Locate, load, and template-sync the config file.

        Copies the bundled template to ``path`` and raises if the file is
        missing (so a first run writes a starter config), then parses it and
        reconciles its key order against the template.

        Args:
            path (str): Path to the config file.
            arr (Arr): Which Arr is being run; selects the arr-prefixed keys.
        """

        template_path = _template_path()
        if not os.path.exists(path):
            shutil.copy(template_path, path)
            raise FileNotFoundError(f"{path} not found. Copying template")

        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls(path=path, arr=arr, data=data)
        config._sync_with_template(template_path)
        return config

    def for_arr(self, arr: Arr) -> "AppConfig":
        """A view of this already-loaded config under a different arr selector.

        The parsed ``data`` is shared with this instance (the file is read and
        template-synced once), so the composition root can load the config a
        single time and hand each arr its own view rather than re-reading and
        re-syncing the same file once per arr. Only the ``{arr}_``-prefixed
        properties differ between views; every global/resolver setting is
        identical. ``data`` is read-only after ``load``, so sharing it is safe.
        """

        return AppConfig(path=self.path, arr=arr, data=self.data)

    def _sync_with_template(self, template_path: str) -> None:
        """Rewrite the config in template key order, inheriting existing values.

        If the file's keys aren't in the template's order (e.g. a new release
        added keys), rebuild it from the template — keeping the user's values
        where present, filling defaults otherwise — and persist the result.
        """

        with open(template_path) as f:
            # Narrow the parsed template mapping (genuinely open YAML our code
            # only reads by key) so the downstream keys/index reads are typed.
            config_template: dict[str, Any] = YAML().load(f)

        if list(self.data.keys()) != list(config_template.keys()):
            # Start from the template (template key order + any newly added
            # defaults) and overwrite with the user's existing values. Keys the
            # user has that the template lacks are intentionally dropped.
            new_config = copy.deepcopy(config_template)
            for key in config_template:
                if key in self.data:
                    new_config[key] = copy.deepcopy(self.data[key])

            self.data = new_config

            with open(self.path, "w+") as f:
                YAML().dump(self.data, f)

    def checksum(self) -> str:
        """MD5 hex digest of the config file's current bytes (cache descriptor)."""

        with open(self.path, "rb") as f:
            return md5(f.read()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """Read a raw config value (for keys without a dedicated property)."""

        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        """Read a config value that must be present and truthy, or raise.

        Mirrors ``get(key)`` but raises ``ValueError`` (naming the key and the
        config path) when the value is missing or empty - the check both arr
        strategies repeat for their required ``*_url`` / ``*_api_key`` settings.
        """

        value = self.data.get(key, None)
        if not value:
            raise ValueError(f"{key} needs to be defined in {self.path}")
        return value

    def _require_str(self, key: str) -> str:
        """``require(key)`` narrowed to ``str`` for the *_url/*_api_key keys.

        The required connection settings are always strings; this wraps the
        generic ``require`` so the typed properties expose ``str`` rather than
        leaking ``Any`` to the arr strategies.
        """

        value = self.require(key)
        assert isinstance(value, str)
        return value

    # --- Typed settings -----------------------------------------------------

    @property
    def sonarr_url(self) -> str:
        return self._require_str("sonarr_url")

    @property
    def sonarr_api_key(self) -> str:
        return self._require_str("sonarr_api_key")

    @property
    def radarr_url(self) -> str:
        return self._require_str("radarr_url")

    @property
    def radarr_api_key(self) -> str:
        return self._require_str("radarr_api_key")

    @property
    def radarr_url_optional(self) -> str | None:
        return self.data.get("radarr_url", None)

    @property
    def radarr_api_key_optional(self) -> str | None:
        return self.data.get("radarr_api_key", None)

    @property
    def ignore_movies_in_radarr(self) -> bool:
        return self.data.get("ignore_movies_in_radarr", False)

    @property
    def ignore_unmonitored(self) -> bool:
        return self.data.get(f"{self.arr}_ignore_unmonitored", False)

    @property
    def qbit_info(self) -> dict[str, Any] | None:
        # The qBittorrent connection-kwargs bag, splatted as
        # ``qbittorrentapi.Client(**qbit_info)`` at the composition root. The
        # values are all strings (host/username/password), but the splat target
        # is a precisely-typed third-party constructor whose keyword params
        # include bools; a ``dict[str, str]`` makes that ``**`` splat a static
        # type error against those bool params. ``Any`` keeps the genuinely
        # dynamic constructor-splat boundary checkable (see Phase 2 follow-up).
        return self.data.get("qbit_info", None)

    @property
    def ignore_seadex_update_times(self) -> bool:
        return self.data.get("ignore_seadex_update_times", False)

    @property
    def use_torrent_hash_to_filter(self) -> bool:
        return self.data.get("use_torrent_hash_to_filter", False)

    @property
    def torrent_category(self) -> str | None:
        return self.data.get(f"{self.arr}_torrent_category", None)

    @property
    def torrent_tags(self) -> list[str] | None:
        return self.data.get("torrent_tags", None)

    @property
    def max_torrents_to_add(self) -> int | None:
        return self.data.get("max_torrents_to_add", None)

    # --- Wait-for-completion + Sonarr manual import -------------------------

    @property
    def import_wait_mode(self) -> ImportWaitMode:
        """How (and whether) to wait for downloads and drive Sonarr import.

        ``off`` (the default) disables the whole feature: no pending records,
        no waiting, no manual import. The other modes (``deferred``,
        ``blocking``, ``hybrid``) control *when* the wait/import runs. An
        unrecognized value falls back to ``ImportWaitMode.OFF`` rather than
        raising, so a typo in the config can never crash a run.
        """

        raw = self.data.get("import_wait_mode", "off")
        try:
            return ImportWaitMode(raw)
        except ValueError:
            return ImportWaitMode.OFF

    @property
    def import_wait_timeout(self) -> int:
        """Seconds to block per torrent in the end-of-run blocking pass.

        A present-but-blank YAML value parses to ``None`` (the key is present, so
        a plain ``.get`` default wouldn't apply); coalesce it to the documented
        default, mirroring the list knobs, so a blanked-out value can't feed
        ``None`` into the wait-loop's ``time.sleep`` / deadline math.
        """

        value = self.data.get("import_wait_timeout")
        return value if value is not None else 3600

    @property
    def import_poll_interval(self) -> int:
        """Seconds between qBittorrent completion polls while waiting.

        Blank coalesces to the default; see :meth:`import_wait_timeout`.
        """

        value = self.data.get("import_poll_interval")
        return value if value is not None else 30

    @property
    def import_ready_timeout(self) -> int:
        """Seconds to wait for Sonarr to import a completed download.

        After qBittorrent reports the torrent complete, the blocking pass asks
        Sonarr to rescan and then polls its queue until the files import (Sonarr's
        own import, or our authoritative manual import on an ``importBlocked``).
        This bounds that second phase; the download wait itself is bounded
        separately by ``import_wait_timeout``. Blank coalesces to the default;
        see :meth:`import_wait_timeout`.
        """

        value = self.data.get("import_ready_timeout")
        return value if value is not None else 600

    @property
    def import_mode(self) -> str:
        """Sonarr ``importMode`` for the manual import (``auto``/``move``/``copy``).

        Blank coalesces to the default; see :meth:`import_wait_timeout`.
        """

        value = self.data.get("import_mode")
        return value if value is not None else "auto"

    @property
    def import_default_quality(self) -> str | None:
        """Fallback quality name when neither our parse nor Sonarr's is known.

        Recommended on a 4K instance (e.g. ``Bluray-2160p``) to avoid
        importing files as Unknown quality and triggering re-grabs.
        """

        return self.data.get("import_default_quality", None)

    @property
    def import_languages_dual(self) -> list[str]:
        """Languages applied to imported files from dual-audio releases.

        A blank YAML value parses to ``None`` (the key is present, so a plain
        ``.get`` default wouldn't apply); coalesce it to the documented default,
        mirroring ``ignore_tags`` / ``trackers``.
        """

        value = self.data.get("import_languages_dual")
        return value if value else ["Japanese", "English"]

    @property
    def import_languages_single(self) -> list[str]:
        """Languages applied to imported files from single-audio releases.

        Blank coalesces to the default; see :meth:`import_languages_dual`.
        """

        value = self.data.get("import_languages_single")
        return value if value else ["Japanese"]

    @property
    def import_pending_max_age_days(self) -> int:
        """Drop pending-import records older than this many days (TTL).

        Blank coalesces to the default; see :meth:`import_wait_timeout`.
        """

        value = self.data.get("import_pending_max_age_days")
        return value if value is not None else 14

    @property
    def discord_url(self) -> str | None:
        return self.data.get("discord_url", None)

    @property
    def public_only(self) -> bool:
        return self.data.get("public_only", True)

    @property
    def prefer_dual_audio(self) -> bool:
        return self.data.get("prefer_dual_audio", True)

    @property
    def want_best(self) -> bool:
        return self.data.get("want_best", True)

    @property
    def ignore_tags(self) -> list[str]:
        ignore_tags = self.data.get("ignore_tags", None)
        if ignore_tags is None:
            ignore_tags = list[str]()
        return ignore_tags

    # cached_property, not property: these two normalize into a fresh ``set`` on
    # every access, and after Phase 5b removed the SeaDexArr mirror attributes
    # they're read inside hot loops (get_seadex_dict / add_torrent). Caching keeps
    # them parse-once per instance. Safe because ``data`` is only ever reassigned
    # by ``_sync_with_template`` during ``load`` (before any property access) and
    # is never mutated afterwards.
    @cached_property
    def ignore_anilist_ids(self) -> set[int]:
        ignore_anilist_ids = self.data.get("ignore_anilist_ids", None)
        if ignore_anilist_ids is None:
            ignore_anilist_ids = set[int]()
        return {int(x) for x in ignore_anilist_ids}

    @cached_property
    def trackers(self) -> set[str]:
        trackers = self.data.get("trackers", None)
        # Default to all trackers (public + private) when none configured.
        # Include private even when public_only: they're filtered later, after
        # the overlap check against what's already downloaded.
        if trackers is None:
            trackers = PUBLIC_TRACKERS.union(PRIVATE_TRACKERS)
        return {t.casefold() for t in trackers}

    @property
    def sleep_time(self) -> int:
        return self.data.get("sleep_time", 2)

    @property
    def cache_time(self) -> int:
        return self.data.get("cache_time", 1)

    @property
    def interactive(self) -> bool:
        return self.data.get("interactive", False)

    @property
    def log_level(self) -> str:
        return self.data.get("log_level", "INFO")

    @property
    def anime_mappings_cfg(self) -> AnimeIdsMap | bool | None:
        return self.data.get("anime_mappings", None)

    @property
    def anidb_mappings_cfg(self) -> ElementTree.Element | bool | None:
        return self.data.get("anidb_mappings", None)

    @property
    def anibridge_mappings_cfg(self) -> AniBridgeGraph | bool | None:
        return self.data.get("anibridge_mappings", None)
