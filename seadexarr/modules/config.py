"""Application configuration: file lifecycle and typed settings.

``AppConfig`` owns everything to do with the YAML config *file* — copying the
bundled template when it's missing, loading it, and keeping its key order in
sync with the template — and exposes the individual settings as typed, normalized
properties so the rest of the package reads ``config.public_only`` instead of
``config.get("public_only", True)`` scattered across the codebase.

Extracted from ``SeaDexArr.__init__`` / ``verify_config`` / ``setup_cache`` in
Phase 2 of the refactor (see ``REFACTOR_PLAN.md``); behaviour-preserving.
"""

import copy
import os
import shutil
from dataclasses import dataclass
from hashlib import md5
from typing import Any

import yaml
from ruamel.yaml import YAML

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
    arr: str
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str, arr: str) -> "AppConfig":
        """Locate, load, and template-sync the config file.

        Copies the bundled template to ``path`` and raises if the file is
        missing (so a first run writes a starter config), then parses it and
        reconciles its key order against the template.

        Args:
            path (str): Path to the config file.
            arr (str): Which Arr is being run ("sonarr"/"radarr"); selects the
                arr-prefixed keys.
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

    def _sync_with_template(self, template_path: str) -> None:
        """Rewrite the config in template key order, inheriting existing values.

        If the file's keys aren't in the template's order (e.g. a new release
        added keys), rebuild it from the template — keeping the user's values
        where present, filling defaults otherwise — and persist the result.
        """

        with open(template_path) as f:
            config_template = YAML().load(f)

        if list(self.data.keys()) != list(config_template.keys()):
            new_config = copy.deepcopy(config_template)
            for key in config_template:
                if key in self.data:
                    new_config[key] = copy.deepcopy(self.data[key])
                else:
                    new_config[key] = copy.deepcopy(config_template[key])

            self.data = copy.deepcopy(new_config)

            with open(self.path, "w+") as f:
                YAML().dump(self.data, f)

    def checksum(self) -> str:
        """MD5 hex digest of the config file's current bytes (cache descriptor)."""

        with open(self.path, "rb") as f:
            return md5(f.read()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """Read a raw config value (for keys without a dedicated property)."""

        return self.data.get(key, default)

    # --- Typed settings -----------------------------------------------------

    @property
    def ignore_unmonitored(self) -> bool:
        return self.data.get(f"{self.arr}_ignore_unmonitored", False)

    @property
    def qbit_info(self) -> dict | None:
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
    def torrent_tags(self) -> list | None:
        return self.data.get("torrent_tags", None)

    @property
    def max_torrents_to_add(self) -> int | None:
        return self.data.get("max_torrents_to_add", None)

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
    def ignore_tags(self) -> list:
        ignore_tags = self.data.get("ignore_tags", None)
        if ignore_tags is None:
            ignore_tags = []
        return ignore_tags

    @property
    def ignore_anilist_ids(self) -> set[int]:
        ignore_anilist_ids = self.data.get("ignore_anilist_ids", None)
        if ignore_anilist_ids is None:
            ignore_anilist_ids = set()
        return {int(x) for x in ignore_anilist_ids}

    @property
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
    def anime_mappings_cfg(self) -> Any:
        return self.data.get("anime_mappings", None)

    @property
    def anidb_mappings_cfg(self) -> Any:
        return self.data.get("anidb_mappings", None)

    @property
    def anibridge_mappings_cfg(self) -> Any:
        return self.data.get("anibridge_mappings", None)
